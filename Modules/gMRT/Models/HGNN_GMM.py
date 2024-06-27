import sys

import torch.nn as nn
import torch
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

from ..gmrt_base import gMRTBase

sys.path.append("../..")
from gnn_utils import InteractionGNNCell, HierarchicalGNNCell, HierarchicalGNNCell_super, DynamicGraphConstruction
from utils import make_mlp, match_dims

from time import time
from torch_geometric.data import Data
import csv

class InteractionGNNBlock(nn.Module):

    """
    An interaction network for embedding class
    """

    def __init__(self, hparams, iterations):
        super().__init__()
            
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """                 
        
        # Setup input network
        self.node_encoder = make_mlp(
            hparams["spatial_channels"],
            hparams["hidden"],
            hparams["latent"],
            hparams["nb_node_layer"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
        )

        # The edge network computes new edge features from connected nodes
        self.edge_encoder = make_mlp(
            2 * (hparams["spatial_channels"]),
            hparams["hidden"],
            hparams["latent"],
            hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
        )

        # Initialize GNN blocks
        if hparams["share_weight"]:
            cell = InteractionGNNCell(hparams)
            ignn_cells = [
                cell
                for _ in range(iterations)
            ]
        else:
            ignn_cells = [
                InteractionGNNCell(hparams)
                for _ in range(iterations)
            ]
        
        self.ignn_cells = nn.ModuleList(ignn_cells)
        
        # output layers
        self.output_layer = make_mlp(
            hparams["latent"],
            hparams["hidden"],
            hparams["emb_dim"],
            hparams["output_layers"],
            layer_norm=hparams["layernorm"],
            output_activation= None,
            hidden_activation=hparams["hidden_output_activation"],
        )
        
        self.hparams = hparams
        
    def forward(self, x, graph):
        
        x.requires_grad = True
        
        nodes = checkpoint(self.node_encoder, x)
        edges = checkpoint(self.edge_encoder, torch.cat([x[graph[0]], x[graph[1]]], dim=1))
        
        for layer in self.ignn_cells:
            nodes, edges= layer(nodes, edges, graph)
        
        embeddings = self.output_layer(nodes)
        embeddings = nn.functional.normalize(embeddings) 
        
        return embeddings, nodes, edges
    
class HierarchicalGNNBlock(nn.Module):

    """
    An hierarchical GNN class
    """

    def __init__(self, hparams, logging):
        super().__init__()
            
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """                
        self.data_dir = hparams["data_dir"]
        #self.super_dir = hparams["super_dir"]
        self.read_counter = 0
        self.write_counter = 0
        self.create_dset = False
 
        self.supernode_encoder = make_mlp(
            hparams["latent"],
            hparams["hidden"],
            hparams["latent"] - hparams["emb_dim"],
            hparams["nb_node_layer"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
        )

        # The edge network computes new edge features from connected nodes
        self.superedge_encoder = make_mlp(
            2 * hparams["latent"],
            hparams["hidden"],
            hparams["latent"],
            hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
        )


        # Initialize GNN blocks
        if hparams["share_weight"]:
            cell = HierarchicalGNNCell(hparams)
            hgnn_cells = [
                cell
                for _ in range(hparams["n_hierarchical_graph_iters"])
            ]
        else:
            hgnn_cells = [
                HierarchicalGNNCell(hparams)
                for _ in range(hparams["n_hierarchical_graph_iters"])
            ]
        
        self.hgnn_cells = nn.ModuleList(hgnn_cells)
        
        # output layers
        self.GMM_model = GaussianMixture(n_components = 2)
        self.super_graph_construction = DynamicGraphConstruction("sigmoid", hparams)
        self.bipartite_graph_construction = DynamicGraphConstruction("exp", hparams)
        self.register_buffer("score_cut", torch.tensor([float("inf")]))
        
        self.log = logging
        self.hparams = hparams

        # Initialize timer variables
        self.cluster_time = 0.
        self.center_time = 0.
        self.construct_time = 0.
        self.graph_init_time = 0.
        self.data_write_time = 0.
        self.preprocess_time = 0.
        self.layer_time = 0.

        # Epoch pooling and graph construction time
        self.epoch_pooling_time = 0.
        self.epoch_graph_construct_time = 0.
  
        # Set profiling flag
        self.profiling = False #True
        if self.profiling:
          self.fd = open('training_supergraph_dims.csv','w')
          header = ['Mean', 'Node', 'Edge', 'Supernode', 'Superedge']
          self.writer = csv.DictWriter(self.fd, fieldnames = header)
          self.writer.writeheader()
    
    def determine_cut(self, cut0):
        """
        determine the cut by requiring the log-likelihood of the edge to belong to the right distribution to be r times higher
        than the log-likelihood to belong to the left. Note that the solution might not exist.
        """
        sigmoid = lambda x: 1/(1+np.exp(-x))
        func = lambda x: sigmoid(self.hparams["cluster_granularity"])*self.GMM_model.predict_proba(x.reshape((-1, 1)))[:, self.GMM_model.means_.argmin()] - sigmoid(-self.hparams["cluster_granularity"])*self.GMM_model.predict_proba(x.reshape((-1, 1)))[:, self.GMM_model.means_.argmax()]
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
            
      
    def clustering(self, x, embeddings, graph):
        if self.profiling:
          cluster_time = time()
        with torch.no_grad():
            if self.profiling:
              likelihood_time = time()
            # Compute cosine similarity transformed by archypertangent
            likelihood = torch.einsum('ij,ij->i', embeddings[graph[0]], embeddings[graph[1]])
            likelihood = torch.atanh(torch.clamp(likelihood, min=-1+1e-7, max=1-1e-7))
            if self.profiling:
              likelihood_time = time() - likelihood_time
              edge_cut_time = time()
            
            # GMM edge cutting
            self.GMM_model.fit(likelihood.unsqueeze(1).cpu().numpy())
            
            # in the case of score cut not initialized, initialize it from the middle point of the two distribution
            if self.score_cut == float("inf"): 
                self.score_cut = torch.tensor([self.GMM_model.means_.mean().item()], device = self.score_cut.device)
    
            cut = self.determine_cut(self.score_cut.item())
            if self.profiling:
              edge_cut_time = time() - edge_cut_time
              exp_avg_time = time()
            
            # Exponential moving average for score cut
            momentum = 0.95
            if self.training & (cut < self.GMM_model.means_.max().item()) & (cut > self.GMM_model.means_.min().item()):
                self.score_cut = momentum*self.score_cut + (1-momentum)*cut
            # In the case the solution if not found, try again from the middle point of the two distribution
            else:
                cut = self.determine_cut(self.GMM_model.means_.mean().item())
                if self.training & (cut < self.GMM_model.means_.max().item()) & (cut > self.GMM_model.means_.min().item()):
                    self.score_cut = momentum*self.score_cut + (1-momentum)*cut
            if self.profiling:
              exp_avg_time = time() - exp_avg_time
            
            self.log("score_cut", self.score_cut.item())
            
            if self.profiling:
              conn_comp_time = time()
            # Connected Components
            mask = likelihood >= self.score_cut.to(likelihood.device)
            #print("mask dim = ", len(mask))
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

            if self.profiling:
              conn_comp_time = time() - conn_comp_time 
              cluster_time = time() - cluster_time
              print("Likelihood Time                 = ",(likelihood_time/cluster_time)*100,"%")
              print("Edge Cutting Time               = ",(edge_cut_time/cluster_time)*100,"%")
              print("Exponential Moving Average Time = ",(exp_avg_time/cluster_time)*100,"%")
              print("Connected Component Time        = ",(conn_comp_time/cluster_time)*100,"%")
              print("Total Cluster Time              = ", cluster_time)
            return clusters

    def save_supergraph_data(self, x, graph, super_graph, supernodes, superedges, batch):
        # Save preprocessed graphs to data directory
        super_dir = self.super_dir
        filename = str(self.write_counter)
        filepath = super_dir + '/train/' + filename
        print("Filepath = ", filepath)
   


        # Save newly generated super graph & corresponding data to file
        input_dict = {}
        event_x_feats = ['x', 'cell_data', 'pid', 'hid', 'pt', \
                         'modulewise_true_edges', 'signal_true_edges']
        event_y_feats = ['y', 'y_pid']
        #super_dict = {'edge_index': super_graph} # edge_index = directed_graph
        super_dict = {'super_graph': super_graph, 'supernodes': supernodes, 'superedges': superedges} 

        for k, v in batch:
          if (k in event_x_feats): # matches x dim
            input_dict[k] = v
          elif k == 'event_file': # exception for event name string
            input_dict[k] = v[0]
          elif (k in event_y_feats): # matches y dim 
            input_dict[k] = v[super_graph[0]]

        # Combine new data & processed old data 
        data = {**super_dict, **input_dict}
        for k, v in data.items():
          if torch.is_tensor(v):
            data[k] = v.clone().detach()
          else:
            data[k] = v

        # Save data to new input directory
        data = Data(**data)
        torch.save(data, filepath)
        self.write_counter += 1

        return

    def preprocess_graphs(self, x, embeddings, nodes, edges, graph, pid, batch):
        
        x.requires_grad = True
        
        # Compute clustering
        if self.profiling:
          cluster_time = time()
        clusters = self.clustering(x, embeddings, graph)
        if self.profiling:
          cluster_time = time() - cluster_time

        # Compute Centers
        if self.profiling:
          center_time = time()
        means = scatter_mean(embeddings[clusters >= 0], clusters[clusters >= 0], dim=0, dim_size=clusters.max()+1)
        means = nn.functional.normalize(means)
        if self.profiling:
          center_time = time() - center_time
        
        # Construct Graphs
        if self.profiling:
          construct_time = time()
        print("graph size = ", graph.size())
        super_graph, super_edge_weights, super_idxs = self.super_graph_construction(means, means, sym = True, norm = True, k = self.hparams["supergraph_sparsity"])
        bipartite_graph, bipartite_edge_weights, bipartite_edge_weights_logits = self.bipartite_graph_construction(embeddings, means, sym = False, norm = True, k = self.hparams["bipartitegraph_sparsity"], logits = True)
        if self.profiling:
          construct_time = time() - construct_time
        
        self.log("clusters", len(means))
        
        # Initialize supernode & edges by aggregating node features. Normalizing with 1-norm to improve training stability

        if self.profiling:
          graph_init_time = time()
        supernodes = scatter_add((nn.functional.normalize(nodes, p=1)[bipartite_graph[0]])*bipartite_edge_weights, bipartite_graph[1], dim=0, dim_size=means.shape[0])
        supernodes = torch.cat([means, checkpoint(self.supernode_encoder, supernodes)], dim = -1)
        superedges = checkpoint(self.superedge_encoder, torch.cat([supernodes[super_graph[0]], supernodes[super_graph[1]]], dim=1))

        if self.profiling:
          graph_init_time = time() - graph_init_time
          data_write_time = time()
        
        # Save preprocessed graphs to data directory
        #self.save_supergraph_data(x, graph, super_graph, supernodes, superedges, batch)

        if self.profiling:
          data_write_time = time() - data_write_time

          self.cluster_time += cluster_time
          self.center_time += center_time
          self.construct_time += construct_time
          self.graph_init_time += graph_init_time
          self.data_write_time += data_write_time
          self.preprocess_time = self.cluster_time + self.center_time + \
                                 self.construct_time + self.graph_init_time + \
                                 self.data_write_time

          self.epoch_pooling_time += cluster_time + center_time
          self.epoch_graph_construct_time += construct_time + graph_init_time
          print("Hierarchical GNN PREprocessing")
          print("---------------------------------")
          print("Cluster Time =             ", self.cluster_time)
          print("Compute Time =             ", self.center_time)
          print("Construct Time =           ", self.construct_time)
          print("Node Initialization Time = ", self.graph_init_time)
          print("Preprocessing Time = ", self.preprocess_time)
        return supernodes, superedges, bipartite_graph, bipartite_edge_weights, super_graph, super_edge_weights
    
    def forward(self, x, embeddings, nodes, edges, graph, pid, batch):
        
        x.requires_grad = True
        if self.create_dset == True:
          supernodes, superedges, bipartite_graph, bipartite_edge_weights, super_graph, super_edge_weights = self.preprocess_graphs(x, embeddings, nodes, edges, graph, pid, batch)
        else: #-------------------------------------------------------------------
          if self.profiling:
            cluster_time = time()
          clusters = self.clustering(x, embeddings, graph)
          if self.profiling:
            cluster_time = time() - cluster_time
            center_time = time()
          means = scatter_mean(embeddings[clusters >= 0], clusters[clusters >= 0], dim=0, dim_size=clusters.max()+1)
          means = nn.functional.normalize(means)
          if self.profiling:
            center_time = time() - center_time
            construct_time = time()
          super_graph, super_edge_weights, super_idxs = self.super_graph_construction(means, means, sym = True, norm = True, k = self.hparams["supergraph_sparsity"])
          bipartite_graph, bipartite_edge_weights, bipartite_edge_weights_logits = self.bipartite_graph_construction(embeddings, means, sym = False, norm = True, k = self.hparams["bipartitegraph_sparsity"], logits = True)
          if self.profiling:
            construct_time = time() - construct_time
            graph_init_time = time()
          supernodes = scatter_add((nn.functional.normalize(nodes, p=1)[bipartite_graph[0]])*bipartite_edge_weights, bipartite_graph[1], dim=0, dim_size=means.shape[0])
          supernodes = torch.cat([means, checkpoint(self.supernode_encoder, supernodes)], dim = -1)
          superedges = checkpoint(self.superedge_encoder, torch.cat([supernodes[super_graph[0]], supernodes[super_graph[1]]], dim=1))
          if self.profiling:
            graph_init_time = time() - graph_init_time
        
        if self.profiling:
          layer_time = time()

        for layer in self.hgnn_cells:
            nodes, edges, supernodes, superedges = layer(nodes,
                                                         edges,
                                                         supernodes,
                                                         superedges,
                                                         graph,
                                                         bipartite_graph,
                                                         bipartite_edge_weights,
                                                         super_graph,
                                                         super_edge_weights)
        if self.profiling:
          layer_time = time() - layer_time

        if self.profiling:
          self.cluster_time += cluster_time
          self.center_time += center_time
          self.construct_time += construct_time
          self.graph_init_time += graph_init_time
          self.preprocess_time = self.cluster_time + self.center_time + \
                                 self.construct_time + self.graph_init_time
          self.epoch_pooling_time += cluster_time + center_time
          self.epoch_graph_construct_time += construct_time + graph_init_time
          print("Hierarchical GNN PREprocessing")
          print("---------------------------------")
          print("Cluster Time =             ", self.cluster_time)
          print("Compute Time =             ", self.center_time)
          print("Construct Time =           ", self.construct_time)
          print("Node Initialization Time = ", self.graph_init_time)
          print("Preprocessing Time = ", self.preprocess_time)
    
          self.layer_time += layer_time
          print("Hierarchical GNN POSTprocessing")
          print("---------------------------")
          print("Layer Construction Time =  ", self.layer_time)

        return nodes, supernodes, bipartite_graph, pid
    
class gMRT(gMRTBase):

    """
    An interaction network class
    """

    def __init__(self, hparams):
        super().__init__(hparams) 

        self.profiling = False #True
        if self.profiling:
          init_time = time()

        self.data_dir = hparams["data_dir"]

        self.ignn_block = InteractionGNNBlock(hparams, hparams["n_interaction_graph_iters"])
        self.hgnn_block = HierarchicalGNNBlock(hparams, self.log)
        
        self.bipartite_output_layer = make_mlp(
            2 * hparams["latent"],
            hparams["hidden"],
            1,
            hparams["output_layers"],
            layer_norm=hparams["layernorm"],
            output_activation= None,
            hidden_activation=hparams["hidden_output_activation"],
        )

        self.read_counter = 0
        
        if self.profiling:
          self.h_time = 0. 
          self.i_time = 0. 
          self.b_time = 0. 
          self.init_time = time() - init_time
          print("Initialization Time = ", self.init_time)
        
    def forward(self, x, graph, pid, batch):

        x.requires_grad = True
       
        load_dset = True
        if load_dset == False:
          directed_graph = torch.cat([graph, graph.flip(0)], dim = 1)
        else:
          directed_graph = graph 
          
        if self.profiling:
          i_time = time() 
        intermediate_embeddings, nodes, edges = self.ignn_block(x, directed_graph)
        print("node emb dim = ", nodes.size())
        if self.profiling:
          i_time = time() - i_time

        if self.profiling:
          h_time = time() 
        nodes, supernodes, bipartite_graph, pid = self.hgnn_block(x, intermediate_embeddings, nodes, edges, directed_graph, pid, batch) 
        if self.profiling:
          h_time = time() - h_time

        if self.profiling:
          b_time = time() 
        bipartite_scores = torch.sigmoid(
            checkpoint(self.bipartite_output_layer, torch.cat([nodes[bipartite_graph[0]], supernodes[bipartite_graph[1]]], dim = 1))
        ).squeeze()
        if self.profiling:
          b_time = time() - b_time 

        if self.profiling:
          self.i_time += i_time
          self.h_time += h_time
          self.b_time += b_time
          print("Interaction Module Time = ", self.i_time)
          print("Hierarchical Module Time = ", self.h_time)
          print("Bipartite Output Layer Time = ", self.b_time)

        return bipartite_graph, bipartite_scores, intermediate_embeddings, pid
