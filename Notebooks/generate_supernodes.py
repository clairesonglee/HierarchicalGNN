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

def avg_pooling(x, edge_index, cluster_dict):
    invalid_key = -1
    if invalid_key in cluster_dict:
      del cluster_dict[invalid_key]

    subcluster_dict = {}
    sub_x = None
    granularity = 0.25
    n_subclusters = 0
    subcluster_labels = []
    node_indices = []
    for key in cluster_dict.keys():
      cluster = cluster_dict[key]
      cluster_size = len(cluster)
      cluster_coords = x[cluster]
      subcluster_size = math.floor(cluster_size * granularity)
      if subcluster_size > 0:
        n_subclusters += 1
        subcluster_labels.extend(np.repeat(key, subcluster_size))
        groups = np.array_split(cluster_coords, subcluster_size, axis=0)
        averages = [group.mean(dim=0, keepdim=True) for group in groups]
        for i, average in enumerate(averages):
          if sub_x == None:
            sub_x = average
          else:
            sub_x = torch.cat((sub_x, average), dim=0)
      assert len(set(subcluster_labels)) == n_subclusters
      assert len(subcluster_labels) == (sub_x.size())[0]
    print("sub_x dim = ", sub_x.size())
    print("subclusters label dim = ", len(subcluster_labels))
    print("num subclusters = ", n_subclusters)
    print("subclusters labels = ", subcluster_labels)
    return sub_x, subcluster_labels, n_subclusters 

def sub_pooling(x, edge_index, cluster_dict):
    invalid_key = -1
    if invalid_key in cluster_dict:
      del cluster_dict[invalid_key]

    subcluster_dict = {}
    sub_x = None
    granularity = 0.25
    n_subclusters = 0
    subcluster_labels = []
    invalid_node_indices = []
    for key in cluster_dict.keys():
      cluster = cluster_dict[key]
      cluster_size = len(cluster)
      cluster_coords = x[cluster]
      subcluster_size = math.floor(cluster_size * granularity)
      if subcluster_size > 0:
        n_subclusters += 1
        subcluster_labels.extend(np.repeat(key, subcluster_size))
        rand_indices = np.random.choice(cluster_size, subcluster_size, replace=False)
        #rand_indices = np.random.randint(0, cluster_size, subcluster_size)
        rand_vals = cluster_coords[rand_indices]
        print('rand idxs = ', rand_indices, 'cluster = ', cluster, 'ridx type = ', type(rand_indices))
        invalid_nodes = np.delete(np.array(cluster), rand_indices)
        invalid_node_indices.extend(invalid_nodes)
        print("rand vals shape = ", rand_vals.shape) 
        print("subcluster size = ", subcluster_size)
        if sub_x == None:
          sub_x = rand_vals
        else:
          sub_x = torch.cat((sub_x, rand_vals), dim=0)
      assert len(set(subcluster_labels)) == n_subclusters
      assert len(subcluster_labels) == (sub_x.size())[0]
    print("sub_x dim = ", sub_x.size())
    print("subclusters label dim = ", len(subcluster_labels))
    print("num subclusters = ", n_subclusters)
    print("subclusters labels = ", subcluster_labels)

    edge_index = edge_index.cpu()
    mask = ~np.isin(edge_index, invalid_node_indices).any(axis=0)
    sub_edge_index = edge_index[:, mask]
    '''
    node_indices = np.arange(len(subcluster_labels))
    src_edge_mask = np.isin(edge_index[0], node_indices)
    dst_edge_mask = np.isin(edge_index[1], node_indices)
    y_mask = np.logical_and(src_edge_mask, dst_edge_mask)
    src_edges = (edge_index[0])[y_mask]
    dst_edges = (edge_index[1])[y_mask]
    sub_edge_index = torch.vstack((src_edges, dst_edges))
    '''
    print("edge index = ", edge_index.size())
    print("sub edge index = ", sub_edge_index.size())

    return sub_x, subcluster_labels, n_subclusters, sub_edge_index

def clustering(x):
    # Run HDBSCAN/DBSCAN clustering algorithm
    cluster_time = time()
    clusterer = HDBSCAN(min_cluster_size=3, min_samples=3)
    #clusterer = DBSCAN(eps=0.5, min_samples=3)
    cluster_labels = clusterer.fit_predict(x.cpu())
    cluster_time = time() - cluster_time

    # Assign nodes using cluster labels
    cluster_dict = {}
    for i, cluster in enumerate(cluster_labels):
      if cluster not in cluster_dict.keys():
        cluster_dict[cluster] = []
      (cluster_dict[cluster]).append(i)

    n_clusters = len(set(cluster_labels)) - 1
    print ("n_clusters (node len) = ", n_clusters)

    return cluster_labels, cluster_dict, n_clusters

def create_coarse_data(output_path, event_dir):
  filename = 0
  total_cluster_time = 0
  y_distrib = []

  for event_path in event_dir:
    # Load event node and graph data 
    event = torch.load(event_path)
    x, directed_graph = event.x, event.edge_index

    cluster_labels, cluster_dict, n_clusters = clustering(x)

    # Aggregate node coordinate values
    node_len = n_clusters
    
    # Convert cluster_dict to a tensor of cluster indices
    cluster_indices = []
    node_indices = []
    for key, val in cluster_dict.items():
      if key != -1:
        cluster_indices.extend([key] * len(val))
        node_indices.extend(val)
    cluster_indices = (torch.tensor(cluster_indices)).to('cuda:0')
    node_indices = (torch.tensor(node_indices)).to('cuda:0')

    # Gather the node coordinates corresponding to the cluster indices
    node_coords = x[node_indices]

    # Compute the mean coordinates for each cluster using scatter_mean
    mean_coords = scatter_mean(node_coords, cluster_indices, dim=0)
    print("node coord dim = ", node_coords.size())
    print("mean coord dim = ", mean_coords.size())


    # Apply mask to event graph features
    pid = scatter_mean(event.pid[node_indices], cluster_indices, dim=0)
    hid = scatter_mean(event.hid[node_indices], cluster_indices, dim=0)
    pt = scatter_mean(event.pt[node_indices], cluster_indices, dim=0)
    cell_data = scatter_mean((event.cell_data)[node_indices,:], cluster_indices, dim=0)
    print("pid = ", pid.size(), "hid = ", hid.size(), "pt = ", pt.size(), "cell data = ", cell_data.size())

    # Remove edges connected to invalid clusters
    #invalid_nodes = cluster_dict[-1]
    event = event.cpu()
    node_indices = node_indices.cpu()
    edge_feats = ['edge_index', 'modulewise_true_edges', 'signal_true_edges']
    src_edge_mask = np.isin(event.edge_index[0], node_indices)
    dst_edge_mask = np.isin(event.edge_index[1], node_indices)
    y_mask = np.logical_and(src_edge_mask, dst_edge_mask)
    src_edges = (event.edge_index[0])[y_mask]
    dst_edges = (event.edge_index[1])[y_mask]
    edges = torch.vstack((src_edges, dst_edges))
    y_true_count = sum(y_mask)
    y = (event.y)[y_mask]
    y_pid = (event.y_pid)[y_mask]
    print("edge dim = ", edges.size(), "# of unique edges = ", len(edges.unique()))
    
    src_edge_mask = np.isin(event.modulewise_true_edges[0], node_indices)
    dst_edge_mask = np.isin(event.modulewise_true_edges[1], node_indices)
    y_mask = np.logical_and(src_edge_mask, dst_edge_mask)
    src_edges = (event.modulewise_true_edges[0])[y_mask]
    dst_edges = (event.modulewise_true_edges[1])[y_mask]
    modwise_edges = torch.vstack((src_edges, dst_edges))

    src_edge_mask = np.isin(event.signal_true_edges[0], node_indices)
    dst_edge_mask = np.isin(event.signal_true_edges[1], node_indices)
    y_mask = np.logical_and(src_edge_mask, dst_edge_mask)
    src_edges = (event.signal_true_edges[0])[y_mask]
    dst_edges = (event.signal_true_edges[1])[y_mask]
    signal_edges = torch.vstack((src_edges, dst_edges))
    print("modwise edges = ", modwise_edges.size(), "signal edges = ", signal_edges.size())

    # Build data dictionary and save to file
    coarse_dict = {'x': mean_coords, 'edge_index': edges, 'y': y, 'y_pid': y_pid, \
                   'cell_data': cell_data, 'pid': pid, 'hid': hid, 'pt': pt, \
                   'modulewise_true_edges': modwise_edges, 'signal_true_edges': signal_edges} 
                   # edge_index = directed_graph
    #print("y distribution = ", y.unique(return_counts=True))
    print("y pid distribution = ", y_pid.unique(return_counts=True))
    print("y distribution = ", y.unique(return_counts=True))
    _, counts = y.unique(return_counts=True)
    ratio = counts[0]/counts[1]
    y_distrib.append(ratio)
    '''
    input_dict = {}
    for feature in event_x_feats:
      input_dict[feature] = event[feature]
    input_dict['event_file'] = event.event_file[0]

    # Combine new data & processed old data 
    data = {**coarse_dict, **input_dict}
    filename = save_data(data, output_path, filename)
    '''
    #filename = save_data(coarse_dict, output_path, filename)
  print("y distribution array = ", y_distrib)  

  # Profile cluster time
  print("Total cluster time = ", total_cluster_time)
  return 

def save_data(data, output_path, filename):
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

def create_super_data(input_path, super_path, hparams):
  super_graph_construction = DynamicGraphConstruction("sigmoid", hparams)
  bipartite_graph_construction = DynamicGraphConstruction("exp", hparams)
  supernode_encoder = make_mlp(
            hparams["latent"],
            hparams["hidden"],
            hparams["latent"] - hparams["emb_dim"],
            hparams["nb_node_layer"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
        )
  self.superedge_encoder = make_mlp(
            2 * hparams["latent"],
            hparams["hidden"],
            hparams["latent"],
            hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
        )

  for event_path in event_dir:
    clusters = clustering(x, graph)

    # Compute Centers
    means = scatter_mean(embeddings[clusters >= 0], clusters[clusters >= 0], dim=0, dim_size=clusters.max()+1)
    means = nn.functional.normalize(means)
        
    # Construct Graphs
    super_graph, super_edge_weights = super_graph_construction(means, means, sym = True, norm = True, k = self.hparams["supergraph_sparsity"])
    bipartite_graph, bipartite_edge_weights, bipartite_edge_weights_logits = bipartite_graph_construction(embeddings, means, sym = False, norm = True, k = hparams["bipartitegraph_sparsity"], logits = True)
        
    # Initialize supernode & edges by aggregating node features. Normalizing with 1-norm to improve training stability
    supernodes = scatter_add((nn.functional.normalize(nodes, p=1)[bipartite_graph[0]])*bipartite_edge_weights, bipartite_graph[1], dim=0, dim_size=means.shape[0])
    supernodes = torch.cat([means, checkpoint(supernode_encoder, supernodes)], dim = -1)
    superedges = checkpoint(superedge_encoder, torch.cat([supernodes[super_graph[0]], supernodes[super_graph[1]]], dim=1))
    print("supernode dim = ", supernodes.size(), "superedge dim = ", superedges.size())


def visualize_data(input_path, super_path, cluster_path):
  event_dir = glob(input_path)
  super_event_dir = glob(super_path)
  cluster_event_dir = glob(cluster_path)
  y_distrib = []
  super_y_distrib = []
  cluster_y_distrib = []
  for i, event_path in enumerate(event_dir):
    # Load event node and graph data 
    event = torch.load(event_path)
    #super_event = torch.load(super_event_dir[i])
    x, y, graph = event.x, event.y, event.edge_index
    if i == 0 or i == 1 or i == 2:
      # Cluster nodes using HDBSCAN
      coords = x.cpu()
      cluster_labels, cluster_dict, n_clusters = clustering(coords)

      '''
      # Create a 3D scatter plot
      fig = plt.figure(figsize=(25,25))
      ax = fig.add_subplot(111, projection='3d')

      # Plot each cluster with a different color
      for cluster_id in range(n_clusters):
        cluster_points = coords[cluster_labels == cluster_id]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f'Cluster {cluster_id}', s=10)

      for j in range(graph.size(1)):
        start, end = graph[0][j], graph[1][j]
        ax.plot([coords[start, 0], coords[end, 0]], [coords[start, 1], coords[end, 1]], [coords[start, 2], coords[end, 2]], color='purple', alpha=0.5)
 
      ax.tick_params(axis='x', which='major', width=100)
      ax.tick_params(axis='y', which='major', width=100)
      ax.tick_params(axis='z', which='major', width=100)

      #ax.set_box_aspect([2, 2, 2]) 

      #plt.savefig('3d_scatter_plot.svg', format='svg')
      #plt.savefig('3d_scatter_plot_' + str(i) + '.png')
      #plt.show()
      '''
      # Pool clusters into subclusters and redraw graph 
      sub_x, subcluster_labels, n_subclusters, sub_graph = sub_pooling(coords, graph, cluster_dict)

      # Create a 3D scatter plot
      fig = plt.figure(figsize=(25,25))
      ax1 = fig.add_subplot(111, projection='3d')
      max_cluster_id = max(subcluster_labels)
      for cluster_id in range(max_cluster_id):
        if cluster_id in subcluster_labels:
          print("cluster id in labels / id = ", cluster_id)
          cluster_indices = [idx for idx, val in enumerate(subcluster_labels) if val == cluster_id]
          print("cluster indices = ", cluster_indices)
          cluster_points = sub_x[cluster_indices,:]
          print("cluster points dim = ", cluster_points.size())
          if cluster_points.dim() == 1:
            cluster_points = cluster_points.unsqueeze(0)
          print("reshaped cluster points dim = ", cluster_points.size())
          ax1.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f'Cluster {cluster_id}', s=10)

      for j in range(sub_graph.size(1)):
        start, end = (sub_graph[0][j]).item(), (sub_graph[1][j]).item()
        if start > end: 
          temp = start
          start = end
          end = temp
        if start > 0 and end < (sub_x.size())[0]:
          print("start = ", start, "end = ", end)
          print("(sub_x.size())[0] = ", (sub_x.size())[0])
          ax1.plot([sub_x[start, 0], sub_x[end, 0]], [sub_x[start, 1], sub_x[end, 1]], [sub_x[start, 2], sub_x[end, 2]], color='blue', alpha=0.5)
      plt.savefig('sub_3d_scatter_plot_' + str(i) + '.png')

    _, counts = y.unique(return_counts=True)
    ratio = counts[0]/counts[1]
    y_distrib.append(ratio)
  '''
  for i, event_path in enumerate(super_event_dir):
    event = torch.load(event_path)
    x, y, graph = event.x, event.y, event.edge_index
    _, counts = y.unique(return_counts=True)
    ratio = counts[0]/counts[1]
    super_y_distrib.append(ratio)
  for i, event_path in enumerate(cluster_event_dir):
    event = torch.load(event_path)
    x, y, graph = event.x, event.y, event.edge_index
    _, counts = y.unique(return_counts=True)
    ratio = counts[0]/counts[1]
    cluster_y_distrib.append(ratio)
  '''
  #print("y distribution array = ", y_distrib)
  #print("super y distribution array = ", super_y_distrib)
  #print("cluster y distribution array = ", cluster_y_distrib)

def main():
  # Set filepaths and initialize variables 
  #input_path = "/data/FNAL/events/train/*"
  #super_path = "/data/FNAL/processed/train/*"
  #output_path = "/data/FNAL/coarse_events/train/"

  #input_path = "/data/FNAL/events/test/*"
  #super_path = "/data/FNAL/processed_no_emb/test/*"
  #cluster_path = "/data/FNAL/processed/test/*"
  #output_path = "/data/FNAL/coarse_events/test/"

  input_path = "/data/FNAL/events/val/*"
  super_path = "/data/FNAL/processed_no_emb/val/*"
  cluster_path = "/data/FNAL/processed/val/*"
  output_path = "/data/FNAL/coarse_events/val/"

  #'''
  event_dir = glob(input_path)
  #data = create_coarse_data(output_path, event_dir)
  visualize_data(input_path, super_path, cluster_path)
  '''
  
  config_path = "/home/csl782/FNAL/HierarchicalGNN/Modules/gMRT/Configs/HGNN_GMM.yaml"
  with open(config_path) as f:
    hparams = yaml.load(f, Loader=yaml.FullLoader)
    data = create_super_data(input_path, super_path, **hparams)
  visualize_data(input_path, super_path, cluster_path)
  '''
main()


