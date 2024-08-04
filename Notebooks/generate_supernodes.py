import yaml
import math
import sys
import random
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

def create_adjacency_list(nodes, edges):
  adjacency_list = {node: [] for node in nodes}
  for edge in edges.T:
    u, v = edge
    u = u.item()  # Convert tensor to standard type
    v = v.item()  # Convert tensor to standard type
    if u in adjacency_list:
        adjacency_list[u].append(v)
    if v in adjacency_list:
        adjacency_list[v].append(u)
  return adjacency_list

def sample_neighbors(adjacency_list, node, sample_size):
    neighbors = adjacency_list[node]
    #print("neighbor len = ", len(neighbors))
    if len(neighbors) == 0:
        return None
    elif len(neighbors) > sample_size:
        sampled_neighbors = random.sample(neighbors, sample_size)
    else:
        sampled_neighbors = random.choices(neighbors, k=sample_size)
    return sampled_neighbors

def create_sampled_nodes(output_path, event_dir, sample_size=20, iterations=2):
  filename = 0
  y_distrib = []

  for idx, event_path in enumerate(event_dir):
    # Load event node and graph data 
    event = torch.load(event_path)
    x, edge_index = event.x, event.edge_index
    event = event.cpu()
    nodes = np.arange(len(x))
    adjacency_list = create_adjacency_list(nodes, edge_index)
    #print("nodes = ", nodes, "edges = ", edge_index)
    #print("adj list = ", adjacency_list)
    max_num_edges = -1
    subgraph_nodes, subgraph_edges = None, None
    node_indices, edge_indices = None, None
    
    for _ in range(iterations):
        iteration_graphs = []
        for node in nodes:
            if node in adjacency_list:
                # Sample neighbors
                sampled_neighbors = sample_neighbors(adjacency_list, node, sample_size)
                # Create the subgraph (node and its sampled neighbors)
                if sampled_neighbors is not None:
                  subgraph_nodes = [node] + sampled_neighbors
                  subgraph_edges = [(node, neighbor) for neighbor in sampled_neighbors]
                  for neighbor in sampled_neighbors:
                      for n in adjacency_list[neighbor]:
                          if n in subgraph_nodes and (neighbor, n) not in subgraph_edges and (n, neighbor) not in subgraph_edges:
                              subgraph_edges.append((neighbor, n))
                  iteration_graphs.append((subgraph_nodes, subgraph_edges))
            if len(subgraph_edges) > max_num_edges:
                node_indices = subgraph_nodes
                edge_indices = subgraph_edges
                max_num_edges = len(subgraph_edges)
    print("max num edges = ", max_num_edges)
    print("subgraph nodes = ", node_indices)
    print("subgraph edge = ", edge_indices)

    matching_cols = np.any(np.isin(np.array(edge_index.cpu()), np.array(node_indices)), axis=0)
    print("matching cols = ", matching_cols)
    edge_indices = np.where(matching_cols)[0]
    subedge_feats = []
    edge_feats = ['y', 'y_pid', 'edge_index']
    for feature in edge_feats:
      edge_feat = getattr(event, feature, None)
      print("Feature = ", feature, "Feature size = ", edge_feat.size())
      if edge_feat.dim() > 1:
        subedge_feat = edge_feat[:, edge_indices]
      else:
        subedge_feat = edge_feat[edge_indices]
      print("Feature = ", feature, "Subfeature size = ", subedge_feat.size())
      subedge_feats.append(subedge_feat)

    # Avoid out of index error by eliminating edge_index values > len(pid)
    edge_index = subedge_feats[2]
    if edge_index.max() >= len(node_indices):
      print("Error: `event.edge_index` contains out-of-bounds indices.")
      print(f"Maximum index in `event.edge_index`: {subedge_feats[2].max()}")
      print(f"Size of `mask` along dimension 0: {len(node_indices)}")

      # Filter out columns with out-of-bounds indices
      valid_edge_indices = (edge_index < len(node_indices)).all(0)
      print("valid_edge_indices = ", valid_edge_indices)
      for i, subedge_feat in enumerate(subedge_feats):
        print("Feature = ", edge_feats[i], "Original subfeature size = ", subedge_feat.size())
        if subedge_feat.dim() > 1:
          filtered_subedge_feat = subedge_feat[:, valid_edge_indices]
          subedge_feats[i] = filtered_subedge_feat
        else:
          filtered_subedge_feat = subedge_feat[valid_edge_indices]
          subedge_feats[i] = filtered_subedge_feat
        print("Feature = ", edge_feats[i], "Filtered subfeature size = ", filtered_subedge_feat.size())

      # TEST IF OUT OF INDEX ERROR OCCURS
      mask = torch.zeros(len(node_indices), dtype=torch.bool)
      for i in subedge_feats[0]:
        graph_mask = mask[subedge_feats[2]].all(0)
      print("Passes DataLoader test")

    subedge_true_feats = []
    edge_true_feats = ['modulewise_true_edges', 'signal_true_edges']
    for feature in edge_true_feats:
      edge_true_feat = getattr(event, feature, None)
      print("Feature = ", feature, "Edge true feature size = ", edge_true_feat.size())
      matching_mask = torch.all(torch.isin(edge_true_feat, edge_index.view(-1)), dim=1)
      subedge_true_feat = edge_true_feat[matching_mask]
      print("Feature = ", feature, "Subedge true feature size = ", subedge_true_feat.size())
      subedge_true_feats.append(subedge_true_feat)

    # Apply mask to event graph features
    node_feats = ['x', 'pid', 'hid', 'pt', 'cell_data'] # all have same x dim
    subnode_feats = []
    print("Node indices size = ", len(node_indices))
    for feature in node_feats:
      node_feat = getattr(event, feature, None)
      if node_feat.dim() > 1:
        subnode_feat = node_feat[node_indices,:]
      else:
        subnode_feat = node_feat[node_indices]
      print("Feature = ", feature, "Subfeature size = ", subnode_feat.size())

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
    #filename = save_data(event, coarse_dict, output_path, filename)
    break

def create_coarse_nodes(output_path, event_dir, resolution):
  filename = 0
  y_distrib = []

  for event_path in event_dir:
    # Load event node and graph data 
    event = torch.load(event_path)
    x, edge_index = event.x, event.edge_index
    event = event.cpu()

    n_nodes = x.size(0)
    n_subnodes = int(math.floor(n_nodes * resolution))
    node_indices = torch.tensor(np.random.choice(n_nodes, n_subnodes, replace=False))
    print("node indices = ", node_indices, "node indices size = ", len(node_indices))
    print("x size = ", x.size(), "sub x size = ", (x[node_indices]).size())

    matching_cols = np.any(np.isin(np.array(edge_index.cpu()), np.array(node_indices)), axis=0)
    edge_indices = np.where(matching_cols)[0]
    subedge_feats = []
    edge_feats = ['y', 'y_pid', 'edge_index']
    for feature in edge_feats:
      edge_feat = getattr(event, feature, None)
      print("Feature = ", feature, "Feature size = ", edge_feat.size())
      if edge_feat.dim() > 1:
        subedge_feat = edge_feat[:, edge_indices]
      else:
        subedge_feat = edge_feat[edge_indices]
      print("Feature = ", feature, "Subfeature size = ", subedge_feat.size())
      subedge_feats.append(subedge_feat)

    # Avoid out of index error by eliminating edge_index values > len(pid)
    edge_index = subedge_feats[2]
    if edge_index.max() >= len(node_indices):
      print("Error: `event.edge_index` contains out-of-bounds indices.")
      print(f"Maximum index in `event.edge_index`: {subedge_feats[2].max()}")
      print(f"Size of `mask` along dimension 0: {len(node_indices)}")

      # Filter out columns with out-of-bounds indices
      valid_edge_indices = (edge_index < len(node_indices)).all(0)
      print("valid_edge_indices = ", valid_edge_indices)
      for i, subedge_feat in enumerate(subedge_feats):
        print("Feature = ", edge_feats[i], "Original subfeature size = ", subedge_feat.size())
        if subedge_feat.dim() > 1:
          filtered_subedge_feat = subedge_feat[:, valid_edge_indices]
          subedge_feats[i] = filtered_subedge_feat
        else:
          filtered_subedge_feat = subedge_feat[valid_edge_indices]
          subedge_feats[i] = filtered_subedge_feat
        print("Feature = ", edge_feats[i], "Filtered subfeature size = ", filtered_subedge_feat.size())

      # TEST IF OUT OF INDEX ERROR OCCURS
      mask = torch.zeros(len(node_indices), dtype=torch.bool)
      for i in subedge_feats[0]:
        graph_mask = mask[subedge_feats[2]].all(0)
      print("Passes DataLoader test")

    subedge_true_feats = []
    edge_true_feats = ['modulewise_true_edges', 'signal_true_edges']
    for feature in edge_true_feats:
      edge_true_feat = getattr(event, feature, None)
      print("Feature = ", feature, "Edge true feature size = ", edge_true_feat.size())
      matching_mask = torch.all(torch.isin(edge_true_feat, edge_index.view(-1)), dim=1)
      subedge_true_feat = edge_true_feat[matching_mask]
      print("Feature = ", feature, "Subedge true feature size = ", subedge_true_feat.size())
      subedge_true_feats.append(subedge_true_feat)

    # Apply mask to event graph features
    node_feats = ['x', 'pid', 'hid', 'pt', 'cell_data'] # all have same x dim
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
    # print("Subgraph label distribution = ", y_distrib)

    #break

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

    print("edge idx = ", event.edge_index)
    print("modwise true edges = ", event.modulewise_true_edges)
    print("signal true edges = ", event.signal_true_edges)

    node_indices = None
    for feature in edge_feats:
      edge_feat = getattr(event, feature, None)
      print("Feature = ", feature, "Feature size = ", edge_feat.size())
      if edge_feat.dim() > 1:
        subedge_feat = edge_feat[:, edge_indices]
        node_indices = set(subedge_feat[0]).union(set(subedge_feat[1]))
        node_indices = torch.tensor(list(node_indices), dtype=torch.int64)
        node_indices = node_indices.cpu()
      else:
        subedge_feat = edge_feat[edge_indices]
      print("Feature = ", feature, "Subfeature size = ", subedge_feat.size())
      subedge_feats.append(subedge_feat)

    # Avoid out of index error by eliminating edge_index values > len(pid)
    if subedge_feats[2].max() >= len(node_indices):
      print("Error: `event.edge_index` contains out-of-bounds indices.")
      print(f"Maximum index in `event.edge_index`: {subedge_feats[2].max()}")
      print(f"Size of `mask` along dimension 0: {len(node_indices)}")
      temp_node_len = len(node_indices)

      # Filter out columns with out-of-bounds indices
      valid_indices = (subedge_feats[2] < len(node_indices)).all(0)
      for i, subedge_feat in enumerate(subedge_feats):
        if subedge_feat.dim() > 1:
          subedge_feats[i] = subedge_feat[:, valid_indices]
          node_indices = set(subedge_feat[0]).union(set(subedge_feat[1]))
          node_indices = torch.tensor(list(node_indices), dtype=torch.int64)
          node_indices = node_indices.cpu()
        else:
          subedge_feats[i] = subedge_feat[valid_indices]

      # TEST IF OUT OF INDEX ERROR OCCURS
      node_len = len(node_indices)
      mask = torch.zeros(node_len, dtype=torch.bool)
      for i in subedge_feats[0]:
        graph_mask = mask[subedge_feats[2]].all(0)
    
    subedge_true_feats = []
    edge_index = subedge_feats[2]
    src_nodes = edge_index[0, :]
    dst_nodes = edge_index[1, :]
    for feature in edge_true_feats:
      edge_true_feat = getattr(event, feature, None)
      print("Feature = ", feature, "Edge true feature size = ", edge_feat.size())
      #subedge_true_feat = edge_true_feat[np.isin(edge_true_feat, np.concatenate((src_nodes, dst_nodes)))]
      subedge_true_feat = edge_true_feat[torch.isin(edge_true_feat, torch.cat((src_nodes, dst_nodes)))]
      print("Feature = ", feature, "Subedge true feature size = ", subedge_feat.size())
      subedge_true_feats.append(subedge_true_feat)
    
    # Apply mask to event graph features
    node_feats = ['x', 'pid', 'hid', 'pt', 'cell_data'] # all have same x dim
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

def data_statistics(input_path):
    print("Input Directory = ", input_path)
    event_dir = glob(input_path)
    total_num_nodes, total_num_edges = 0, 0
    num_nodes, num_edges = [], []
    for i, event_path in enumerate(event_dir):
        event = torch.load(event_path)
        x, y, edge_index = event.x, event.y, event.edge_index
        num_node, num_edge = x.size(0), edge_index.size(1)
        num_nodes.append(num_node)
        num_edges.append(num_edge)
        total_num_nodes += num_node
        total_num_edges += num_edge
    print("Total number of nodes = ", total_num_nodes)
    print("Total number of edges = ", total_num_edges)
    print("====================================")

    num_nodes = np.array(num_nodes)
    mean_value = np.mean(num_nodes)
    median_value = np.median(num_nodes)
    std_dev_value = np.std(num_nodes)
    min_value = np.min(num_nodes)
    max_value = np.max(num_nodes)

    print("Node Data Statistics")
    print(f"Mean: {mean_value}")
    print(f"Median: {median_value}")
    print(f"Standard Deviation: {std_dev_value}")
    print(f"Minimum: {min_value}")
    print(f"Maximum: {max_value}")
    print("====================================")

    num_edges = np.array(num_edges)
    mean_value = np.mean(num_edges)
    median_value = np.median(num_edges)
    std_dev_value = np.std(num_edges)
    min_value = np.min(num_edges)
    max_value = np.max(num_edges)

    print("Edge Data Statistics")
    print(f"Mean: {mean_value}")
    print(f"Median: {median_value}")
    print(f"Standard Deviation: {std_dev_value}")
    print(f"Minimum: {min_value}")
    print(f"Maximum: {max_value}")
    print("====================================")

def y_stats(event_dir, subevent_dir):
  y_distrib, sub_y_distrib = [], []
  for i, event_path in enumerate(event_dir):
    event = torch.load(event_path)
    y, y_pid = event.y, event.pid
    # Count true instances in y and y_pid labels
    _, counts = y.unique(return_counts=True)
    ratio = (counts[0]/counts[1]).item()
    y_distrib.append(ratio)  

  for i, subevent_path in enumerate(subevent_dir):
    subevent = torch.load(subevent_path)
    y, y_pid = subevent.y, subevent.pid
    # Count true instances in y and y_pid labels
    _, counts = y.unique(return_counts=True)
    ratio = (counts[0]/counts[1]).item()
    sub_y_distrib.append(ratio)  
  
  y_distrib = np.array(y_distrib)
  mean_value = np.mean(y_distrib)
  median_value = np.median(y_distrib)
  std_dev_value = np.std(y_distrib)
  min_value = np.min(y_distrib)
  max_value = np.max(y_distrib)

  print("Graph Data Statistics")
  print(f"Mean: {mean_value}")
  print(f"Median: {median_value}")
  print(f"Standard Deviation: {std_dev_value}")
  print(f"Minimum: {min_value}")
  print(f"Maximum: {max_value}")
  print("====================================")

  sub_y_distrib = np.array(sub_y_distrib)
  mean_value = np.mean(sub_y_distrib)
  median_value = np.median(sub_y_distrib)
  std_dev_value = np.std(sub_y_distrib)
  min_value = np.min(sub_y_distrib)
  max_value = np.max(sub_y_distrib)

  print("Subgraph Data Statistics")
  print(f"Mean: {mean_value}")
  print(f"Median: {median_value}")
  print(f"Standard Deviation: {std_dev_value}")
  print(f"Minimum: {min_value}")
  print(f"Maximum: {max_value}")

def main():
  # Set filepaths and initialize variables

  input_path = "/data/FNAL/events/train/*"
  output_path = "/data/FNAL/coarse_nodes/10p-res/train/"

  #input_path = "/data/FNAL/events/test/*"
  #output_path = "/data/FNAL/coarse_nodes/10p-res/test/"

  #input_path = "/data/FNAL/events/val/*"
  #output_path = "/data/FNAL/coarse_nodes/10p-res/val/"

  event_dir = glob(input_path)
  #subevent_dir = glob(output_path)
  #y_stats(event_dir, subevent_dir)
  
  resolution = 0.10
  #data = create_coarse_nodes(output_path, event_dir, resolution)
  #visualize_data(input_path, super_path, cluster_path)

  input_path = "/data/FNAL/events/train/*"
  output_path = "/data/FNAL/sampled_nodes/train/"

  sample_size = 5
  iterations = 1
  data = create_sampled_nodes(output_path, event_dir, sample_size, iterations)

  #data_path = "/data/FNAL/coarse_nodes/50p-res/train/*"
  #data_path = "/data/FNAL/events/train/*"
  #data_statistics(data_path)
  #'''
main()


