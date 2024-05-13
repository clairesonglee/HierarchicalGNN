import yaml
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

sys.path.append("..")
#from Modules.gMRT.Models.HGNN_GMM import InteractionGNNBlock, HierarchicalGNNBlock
#from Modules.utils import TrackMLDataset


def determine_cut(self, cut0):
  """
  determine the cut by requiring the log-likelihood of the edge to belong to the right distribution to be r times higher
  than the log-likelihood to belong to the left. Note that the solution might not exist.
  """
  sigmoid = lambda x: 1/(1+np.exp(-x))
  amples=unc = lambda x: sigmoid(self.hparams["cluster_granularity"])*self.GMM_model.predict_proba(x.reshape((-1, 1)))[:, self.GMM_model.means_.argmin()] - sigmoid(-self.hparams["cluster_granularity"])*self.GMM_model.predict_proba(x.reshape((-1, 1)))[:, self.GMM_model.means_.argmax()]
  cut = fsolve(func, cut0)
  return cut.item()

def get_cluster_labels(self, connected_components, x):
        
  clusters = -torch.ones(len(x), device = x.device).long()
  labels = torch.as_tensor(connected_components["labels"], device = x.device)
  vertex = torch.tensor(connected_components["vertex"], device = x.device) 
  _, inverse, counts = labels.unique(return_inverse = True, return_counts = True)
  mask = counts[inverse] >= self.hparams["min_cluster_size"]
  clusters[vertex[mask]] = labels[mask].unique(return_inverse = True)[1].long()
        
  return clusters

def clustering():

  with torch.no_grad():
    
    # Compute cosine similarity transformed by archypertangent
    likelihood = torch.einsum('ij,ij->i', embeddings[graph[0]], embeddings[graph[1]])
    likelihood = torch.atanh(torch.clamp(likelihood, min=-1+1e-7, max=1-1e-7))
    
    # GMM edge cutting
    self.GMM_model.fit(likelihood.unsqueeze(1).cpu().numpy())
    
    # in the case of score cut not initialized, initialize it from the middle point of the two distribution
    if self.score_cut == float("inf"): 
      self.score_cut = torch.tensor([self.GMM_model.means_.mean().item()], device = self.score_cut.device)

    cut = self.determine_cut(self.score_cut.item())
    
    # Exponential moving average for score cut
    momentum = 0.95
    if self.training & (cut < self.GMM_model.means_.max().item()) & (cut > self.GMM_model.means_.min().item()):
      self.score_cut = momentum*self.score_cut + (1-momentum)*cut
    # In the case the solution if not found, try again from the middle point of the two distribution
    else:
      cut = self.determine_cut(self.GMM_model.means_.mean().item())
      if self.training & (cut < self.GMM_model.means_.max().item()) & (cut > self.GMM_model.means_.min().item()):
        self.score_cut = momentum*self.score_cut + (1-momentum)*cut
    
    self.log("score_cut", self.score_cut.item())
    
    # Connected Components
    mask = likelihood >= self.score_cut.to(likelihood.device)
    try:
      G = cugraph.Graph()
      df = cudf.DataFrame({"src": cp.asarray(graph[0, mask]),
		           "dst": cp.asarray(graph[1, mask]),
		         })            
      G.from_cudf_edgelist(df, source = "src", destination = "dst")
      connected_components = cugraph.components.connected_components(G)
      clusters = self.get_cluster_labels(connected_components, x)
      if clusters.max() <= 2:
        raise ValueError
    except ValueError:
      # Sometimes all edges got cut, then it will run into value error. In that case just use the original graph
      G = cugraph.Graph()
      df = cudf.DataFrame({"src": cp.asarray(graph[0]),
		           "dst": cp.asarray(graph[1]),
		         })            
      G.from_cudf_edgelist(df, source = "src", destination = "dst")
      connected_components = cugraph.components.connected_components(G)
      clusters = self.get_cluster_labels(connected_components, x)

    return clusters

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

def create_coarse_data():
  # Set filepaths and initialize variables 
  input_path = "/data/FNAL/events/train/*"
  output_path = "/data/FNAL/coarse_events/train/"
  #input_path = "/data/FNAL/events/test/*"
  #output_path = "/data/FNAL/coarse_events/test/"
  input_path = "/data/FNAL/events/val/*"
  output_path = "/data/FNAL/coarse_events/val/"
  config_path = "/home/csl782/FNAL/HierarchicalGNN/Modules/gMRT/Configs/HGNN_GMM.yaml"
  event_dir = glob(input_path)
  filename = 0
  total_cluster_time = 0

  for event_path in event_dir:
    # Load event node and graph data 
    event = torch.load(event_path)
    x, directed_graph = event.x, event.edge_index

    # Run HDBSCAN/DBSCAN clustering algorithm
    cluster_time = time()
    clusterer = HDBSCAN(min_cluster_size=3, min_samples=3)
    #clusterer = DBSCAN(eps=0.5, min_samples=3)
    cluster_labels = clusterer.fit_predict(x.cpu())
    total_cluster_time += time() - cluster_time

    # Assign nodes using cluster labels
    cluster_dict = {}
    for i, cluster in enumerate(cluster_labels):
      if cluster not in cluster_dict.keys():
        cluster_dict[cluster] = []
      (cluster_dict[cluster]).append(i)

    # Aggregate node coordinate values
    node_len = len(set(cluster_labels)) - 1 # exclude -1 (invalid nodes)
    nodes = None
    for i, (key, val) in enumerate(cluster_dict.items()):
      if key != -1:
        a, b, c = 0, 0, 0
        # Sum node coordinates
        for idx in val:
          a += x[idx][0]
          b += x[idx][1]
          c += x[idx][2]
        # Compute coordinate averages
        num_indices = len(val)
        a, b, c = a/num_indices, b/num_indices, c/num_indices
        avg_coords = torch.tensor([a, b, c])
        if nodes == None:
          nodes = avg_coords.clone().detach()
        else:
          nodes = torch.vstack((nodes, avg_coords)) 
    print("x dim = ", x.size(), "graph dim = ", directed_graph.size())
    print("node dim = ", nodes.size(), "node len = ", node_len)

    # Create mask for node values
    x_len = len(x)
    x_mask = [False] * x_len
    for i, val in enumerate(cluster_dict.keys()):
      if val != -1:
        x_mask[val] = True # x, cell data, pid, hid, pt
    print("x mask len = ", len(x_mask), "x dim = ", len(x), "node dim = ", nodes.size())
    #print("x mask trunc = ", x_mask[:node_len])
    
    # Apply mask to event graph features
    event = event.cpu()
    pid = event.pid[:node_len]
    hid = event.hid[:node_len]
    pt = event.pt[:node_len] 
    cell_data = (event.cell_data)[:node_len, :]
    print("pid = ", pid.size(), "hid = ", hid.size(), "pt = ", pt.size(), "cell data = ", cell_data.size())

    # Remove edges connected to invalid clusters
    #invalid_nodes = cluster_dict[-1]
    valid_nodes = np.arange(len(pid))
    edge_feats = ['edge_index', 'modulewise_true_edges', 'signal_true_edges']
    src_edge_mask = np.isin(event.edge_index[0], valid_nodes)
    dst_edge_mask = np.isin(event.edge_index[1], valid_nodes)
    y_mask = np.logical_and(src_edge_mask, dst_edge_mask)
    src_edges = (event.edge_index[0])[y_mask]
    dst_edges = (event.edge_index[1])[y_mask]
    edges = torch.vstack((src_edges, dst_edges))
    y_true_count = sum(y_mask)
    y = (event.y)[y_mask]
    y_pid = (event.y_pid)[y_mask]
    print("edge dim = ", edges.size(), "# of unique edges = ", len(edges.unique()))

    
    src_edge_mask = np.isin(event.modulewise_true_edges[0], valid_nodes)
    dst_edge_mask = np.isin(event.modulewise_true_edges[1], valid_nodes)
    y_mask = np.logical_and(src_edge_mask, dst_edge_mask)
    src_edges = (event.modulewise_true_edges[0])[y_mask]
    dst_edges = (event.modulewise_true_edges[1])[y_mask]
    modwise_edges = torch.vstack((src_edges, dst_edges))

    src_edge_mask = np.isin(event.signal_true_edges[0], valid_nodes)
    dst_edge_mask = np.isin(event.signal_true_edges[1], valid_nodes)
    y_mask = np.logical_and(src_edge_mask, dst_edge_mask)
    src_edges = (event.signal_true_edges[0])[y_mask]
    dst_edges = (event.signal_true_edges[1])[y_mask]
    signal_edges = torch.vstack((src_edges, dst_edges))
    print("modwise edges = ", modwise_edges.size(), "signal edges = ", signal_edges.size())

    # Build data dictionary and save to file
    coarse_dict = {'x': nodes, 'edge_index': edges, 'y': y, 'y_pid': y_pid, \
                   'cell_data': cell_data, 'pid': pid, 'hid': hid, 'pt': pt, \
                   'modulewise_true_edges': modwise_edges, 'signal_true_edges': signal_edges} 
                   # edge_index = directed_graph
    '''
    input_dict = {}
    for feature in event_x_feats:
      input_dict[feature] = event[feature]
    input_dict['event_file'] = event.event_file[0]

    # Combine new data & processed old data 
    data = {**coarse_dict, **input_dict}
    filename = save_data(data, output_path, filename)
    '''
    filename = save_data(coarse_dict, output_path, filename)

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

def main():
  data = create_coarse_data()

main()


