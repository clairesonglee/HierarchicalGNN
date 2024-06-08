import yaml
import math
import sys
import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_mean
from torch.utils.checkpoint import checkpoint
import numpy as np
import cudf
import cupy as cp
from sklearn.mixture import GaussianMixture
import cugraph
from scipy.optimize import fsolve
import numpy as np
import uuid
from torch.utils.data import random_split
from sklearn.cluster import HDBSCAN 
from sklearn.cluster import DBSCAN 
from glob import glob
from time import time 
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append("..")
#from Modules.gMRT.Models.HGNN_GMM import InteractionGNNBlock, HierarchicalGNNBlock
#from Modules.utils import TrackMLDataset
from Modules.utils import make_mlp

np.set_printoptions(threshold=sys.maxsize)

def create_dataset():
  with open(config_path, 'r') as file:
    params = yaml.safe_load(file)
  paths = load_dataset_paths(hparams["super_dir"], hparams["datatype_names"])
  trainset, valset, testset = random_split(paths, hparams["test_split"], generator=torch.Generator().manual_seed(0))
  trainset = TrackMLDataset(trainset, hparams, stage = "train", device = "cpu")
  valset = TrackMLDataset(valset, hparams, stage = "train", device = "cpu")
  testset = TrackMLDataset(testset, hparams, stage = "train", device = "cpu")
  for event in trainset:
    x, directed_graph = event.x, event.edge_index
    print("x dim = ", x.size(), "graph dim = ", graph.size())

#def subsampling(x, edge_index, cluster_dict):

def create_coarse_data(output_path, event_dir, resolution):
  filename = 0
  y_distrib = []

  for event_path in event_dir:
    # Load event node and graph data 
    event = torch.load(event_path)
    x, edge_index = event.x, event.edge_index
    event = event.cpu()
    #print("Event attr: ", dir(event))

    n_edges = edge_index.size(1)
    n_subedges = int(math.floor(n_edges * resolution))
    subedge_feats = []
    edge_feats = ['y', 'y_pid', 'edge_index']
    edge_true_feats = ['modulewise_true_edges', 'signal_true_edges']
    edge_indices = torch.tensor(np.random.choice(n_edges, n_subedges, replace=False))
    print("Edge indices size = ", len(edge_indices))
    edge_indices = edge_indices.cpu()

    node_indices = None
    for feature in edge_feats:
      edge_feat = getattr(event, feature, None)
      print("Feature = ", feature, "Feature size = ", edge_feat.size())
      if edge_feat.dim() > 1:
        subedge_feat = edge_feat[:, edge_indices]
        node_indices = set(subedge_feat[0]).union(set(subedge_feat[1]))
        node_indices = torch.tensor(list(node_indices), dtype=torch.int64)
      else:
        subedge_feat = edge_feat[edge_indices]
      print("Feature = ", feature, "Subfeature size = ", subedge_feat.size())
      subedge_feats.append(subedge_feat)
    node_indices = node_indices.cpu()
    
    subedge_true_feats = []
    for feature in edge_true_feats:
      edge_true_feat = getattr(event, feature, None)
      print("Feature = ", feature, "Feature size = ", edge_feat.size())
      subedge_true_feat = edge_true_feat
      subedge_true_feats.append(subedge_true_feat)
    
    # Apply mask to event graph features
    node_feats = ['x', 'pid', 'hid', 'pt', 'cell_data']
    subnode_feats = []
    print("Node indices size = ", len(node_indices))
    for feature in node_feats:
      node_feat = getattr(event, feature, None)
      if node_feat.dim() > 1:
        subnode_feat = node_feat[node_indices,:]
      else:
        subnode_feat = node_feat[node_indices]
      print("Feature = ", feature, "Subfeature size = ", subnode_feat.size())
      subnode_feats.append(subnode_feat)


    # Build data dictionary and save to file
    coarse_dict = {'x': subnode_feats[0], \
                   'pid': subnode_feats[1], \
                   'hid': subnode_feats[2], \
                   'pt': subnode_feats[3], \
                   'cell_data': subnode_feats[4], \
                   'y': subedge_feats[0], \
                   'y_pid': subedge_feats[1], \
                   'edge_index': subedge_feats[2], \
                   'modulewise_true_edges': subedge_true_feats[0], \
                   'signal_true_edges': subedge_true_feats[1]}
    filename = save_data(event, coarse_dict, output_path, filename)
    # Count true instances in y and y_pid labels
    y = subedge_feats[0]
    y_pid = subedge_feats[1]
    _, counts = y.unique(return_counts=True)
    ratio = counts[0]/counts[1]
    y_distrib.append(ratio)  
  print("Subgraph label distribution = ", y_distrib)  

  return 

def save_data(event, data, output_path, filename):

    # Combine new data & processed old data 
    #data = {**coarse_dict, **input_dict}
    for k, v in data.items():
      if torch.is_tensor(v):
        data[k] = v.clone().detach()
      else:
        data[k] = v

    # Save data to new input directory
    data = Data(**data)
    output_name = output_path + str(filename)
    print("Path: ", output_name)
    torch.save(data, output_name)
    filename += 1
    return filename

def plot_input(graph, coords, i):
      # Create a 3D scatter plot
      fig = plt.figure(figsize=(25,25))
      ax = fig.add_subplot(111, projection='3d')

      n_edges = graph.size(1)
      print("Edges = ", n_edges)
      for j in range(n_edges):
        start, end = graph[0][j], graph[1][j]
        ax.plot([coords[start, 0], coords[end, 0]], [coords[start, 1], coords[end, 1]], [coords[start, 2], coords[end, 2]], color='black', alpha=0.5)
 
      #ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2])

      ax.tick_params(axis='x', which='major', width=100)
      ax.tick_params(axis='y', which='major', width=100)
      ax.tick_params(axis='z', which='major', width=100)

      plt.savefig('input_plot_' + str(i) + '.png')

def plot_subgraph(graph, coords, idx):
      # Sample edges using hyperparameter
      n_subgraphs = 4
      resolutions = [0.5, 0.25, 0.10, 0.05]
      names = ['50%', '25%', '10%', '5%']
      colors = ['blue', 'green', 'purple', 'red']

      n_edges = graph.size(1)
      for i in range(n_subgraphs):
        # Create a 3D scatter plot
        fig = plt.figure(figsize=(25,25))
        ax1 = fig.add_subplot(111, projection='3d')

        resolution = resolutions[i]
        name = names[i]
        color = colors[i]
        n_subedges = int(math.floor(n_edges * resolution))
        edge_indices = np.random.choice(n_edges, n_subedges, replace=False)
        print("Edges = ", n_edges, "Subedges = ", n_subedges)
        for j in range(n_subedges):
          edge = edge_indices[j]
          start, end = (graph[0][edge]).item(), (graph[1][edge]).item()
          ax1.plot([coords[start, 0], coords[end, 0]], [coords[start, 1], coords[end, 1]], [coords[start, 2], coords[end, 2]], color=color, alpha=0.5)

        #ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2])

        plt.savefig('subsample_plot_' + name + '_' + str(idx) + '.png')
        

def visualize_data(input_path, super_path, cluster_path):
  event_dir = glob(input_path)
  y_distrib = []
  for i, event_path in enumerate(event_dir):
    # Load event node and graph data 
    event = torch.load(event_path)
    x, y, y_pid, graph = event.x, event.y, event.y_pid, event.edge_index
    test_plots = np.arange(2)
    if i in test_plots:
      # Cluster nodes using HDBSCAN
      coords = x.cpu()
      #plot_input(graph, coords, i)
      #plot_subgraph(graph, coords, i)

    # Count true instances in y and y_pid labels
    _, counts = y.unique(return_counts=True)
    ratio = counts[0]/counts[1]
    y_distrib.append(ratio)  
  print("Input label distribution = ", y_distrib)  

def main():
  # Set filepaths and initialize variables 
  #input_path = "/data/FNAL/events/train/*"
  #super_path = "/data/FNAL/processed/train/*"
  #cluster_path = "/data/FNAL/processed/train/*"
  #output_path = "/data/FNAL/coarse_events/10p-res/train/"

  #input_path = "/data/FNAL/events/test/*"
  #super_path = "/data/FNAL/processed_no_emb/test/*"
  #cluster_path = "/data/FNAL/processed/test/*"
  #output_path = "/data/FNAL/coarse_events/10p-res/test/"

  input_path = "/data/FNAL/events/val/*"
  super_path = "/data/FNAL/processed_no_emb/val/*"
  cluster_path = "/data/FNAL/processed/val/*"
  output_path = "/data/FNAL/coarse_events/10p-res/val/"

  #'''
  event_dir = glob(input_path)
  resolution = 0.10
  data = create_coarse_data(output_path, event_dir, resolution)
  #visualize_data(input_path, super_path, cluster_path)
  '''
  config_path = "/home/csl782/FNAL/HierarchicalGNN/Modules/gMRT/Configs/HGNN_GMM.yaml"
  with open(config_path) as f:
    hparams = yaml.load(f, Loader=yaml.FullLoader)
    data = create_super_data(input_path, super_path, **hparams)
  visualize_data(input_path, super_path, cluster_path)
  '''
main()


