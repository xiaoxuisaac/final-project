# GCNs.py - First created by Kincaid MacDonald in Spring 2021.
# Deep Learning Theory and Applications - Assignment 4.
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch.nn import Sequential as Seq, Linear, ReLU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BetterGCNConv(MessagePassing):
    def __init__(self, nodes_channels, edge_channels, message_channels, input_channels = 0):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        
        if input_channels == 0:
            input_channels = nodes_channels
        self.mlp1 = Seq(Linear(input_channels + edge_channels, message_channels),
                       ReLU(),
                       Linear(message_channels, message_channels),
                       ReLU(),
                       Linear(message_channels, message_channels)
                       )

        self.mlp2 = Seq(Linear(message_channels + input_channels,message_channels),
                       ReLU(),
                       Linear(message_channels, message_channels),
                       ReLU(),
                       Linear(message_channels, nodes_channels))


    def forward(self, x, edge_index, edge_attr):
        out =  self.propagate(edge_index, x=x, edge_attr = edge_attr)
                        
        return self.mlp2(torch.cat((x, out), -1))

    def message(self, x_j, edge_attr):
        return self.mlp1(torch.cat((x_j, edge_attr),-1))


def graph_collapse(x, dimq, bz_number, batch_number):
    x = x.view((batch_number, 2*bz_number+1, dimq, -1))
    x = x.sum(1)
    return x.view(batch_number*dimq, -1)


    
class FloquetSolver(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features, edge_features):
        super(FloquetSolver, self).__init__()
        torch.manual_seed(12345)
        
        self.encoder= Seq(Linear(4, num_node_features),
                        ReLU(),
                        Linear(num_node_features, num_node_features))
        
        self.conv1 = BetterGCNConv(num_node_features, edge_features, 
                                   hidden_channels, input_channels=0)        
        self.conv2 = BetterGCNConv(num_node_features, edge_features, hidden_channels)
        self.conv3 = BetterGCNConv(num_node_features, edge_features, hidden_channels)
        self.conv4 = BetterGCNConv(num_node_features, edge_features, hidden_channels)
        self.conv5 = BetterGCNConv(num_node_features, edge_features, hidden_channels)
                
        self.decoder_r = Seq(Linear(num_node_features+1, 256),
                       ReLU(),
                       Linear(256, 64),
                       ReLU(),
                       Linear(64, 1))
        
    def forward(self, x, edge_index, edge_attr, bz_number, dimq, omega_p, batch, root = True):
        omega_p = torch.tensor(omega_p).float()
        x = x.float()
        edge_attr = edge_attr.float()
        
        if batch is not None: 
            dimq = int(dimq[0])
            bz_number = int(bz_number[0])
        else:
            batch = torch.zeros(len(x)).int()
        
        nodes_number = dimq*(2*bz_number +1)
        
        
        batch_number = int(batch[-1]) + 1
        
        # 1-D array to only store the diagonal term
        matrix = torch.zeros((batch_number, dimq)).to(device)
        
        
        
        
        for i in range(dimq):
            
            xi = x.detach().clone()
            root_index = torch.arange(0, len(xi))%nodes_number == bz_number*dimq + i
            
            #label the root nodes.
            if root:
                xi[root_index, 2] = 1             
            
            offsets = xi[root_index,0]
            offsets = offsets.repeat_interleave(nodes_number)
            
            if not root:
                offsets[:] = 0
            
            
            xi = torch.cat((offsets.unsqueeze(-1), xi),1)
            
            
            
            
            xi = self.encoder(xi)
            
            xi = self.conv1(xi, edge_index, edge_attr)
            # # xi = F.dropout(xi, training=self.training)
            xi = self.conv2(xi, edge_index, edge_attr)
            # # xi = F.dropout(xi, training=self.training)
            xi = self.conv3(xi, edge_index, edge_attr)
            # # xi = F.dropout(xi, training=self.training)
            xi = self.conv4(xi, edge_index, edge_attr)
            # # xi = F.dropout(xi, training=self.training)
            xi = self.conv5(xi, edge_index, edge_attr)
            

            xi = torch.cat((offsets.unsqueeze(-1), xi),1)

            #rooted graph gives diagonal entry
            decode = xi.reshape((batch_number, nodes_number, -1))
            decode = decode[:, bz_number*dimq + i, :] #shape (batch_number, hidden_channels)


            
            decode = self.decoder_r(decode)
            matrix[:, i] = decode.view(-1)
            
            
        return matrix.squeeze()
        


    
class FloquetRecurrentSolver(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features, 
                 edge_features, memo = 2):
        super(FloquetSolver, self).__init__()
        torch.manual_seed(12345)
        
        self.encoder= Seq(Linear(4, num_node_features),
                        ReLU(),
                        Linear(num_node_features, num_node_features))
        
        self.conv1 = BetterGCNConv(num_node_features, edge_features, 
                                   hidden_channels, input_channels=num_node_features*memo)        
        self.conv2 = BetterGCNConv(num_node_features, edge_features, hidden_channels)
        self.conv3 = BetterGCNConv(num_node_features, edge_features, hidden_channels)
        self.conv4 = BetterGCNConv(num_node_features, edge_features, hidden_channels)
        self.conv5 = BetterGCNConv(num_node_features, edge_features, hidden_channels)
                
        self.decoder_r = Seq(Linear(num_node_features+1, 256),
                       ReLU(),
                       Linear(256, 64),
                       ReLU(),
                       Linear(64, 1))
                
        
    def forward(self, x, edge_index, edge_attr, bz_number, dimq, omega_p, batch):
        x = x.float()
        edge_attr = edge_attr.float()
        if batch is not None: 
            dimq = int(dimq[0])
            bz_number = int(bz_number[0])
        else:
            batch = torch.zeros(len(x)).int()
            
        bz_number = bz_number
        dimq = dimq
            
        nodes_number = dimq*(2*bz_number +1)
        
        
        batch_number = int(batch[-1]) + 1
        
        # 1-D array to only store the diagonal term
        matrix = torch.zeros((self.batch_number, dimq)).to(device)
        
        
        root_index = x[:,1] == bz_number        
        self.evals = x[root_index, 0]
        
        
            
        x_memo = 0 # final embedding of the nodes. dimension (len(batch), memo*num_node_features)
        
        return matrix
            
        
    def iter_round(self, x, x_memo, edge_index, edge_attr,
                   batch_number, dimq, bz_number, memo):

        nodes_number = (bz_number*2+1)*dimq
        
        
        
        x_memo # (batch_number*dimq, memo*num_node_features)
        
        x_memo = x_memo.view(batch_number, dimq, -1)
        
        x_memo = x_memo.repeat_interleave(2*bz_number+1, 0)
        
        x_memo = x_memo.view(len(x), -1)
        
        end_embedding = torch.zeros((len(x_memo), int(len(x_memo[0])/memo))) 
        # (len(batch), memo*num_node_features)
        
        for i in range(dimq):
            xi = x.detach().clone()
            root_index = torch.arange(0, len(xi))%nodes_number == bz_number*dimq + i
            
            #label the root nodes.
            xi[root_index, 2] = 1             
            
            offsets = xi[root_index,0]
            offsets = offsets.repeat_interleave(nodes_number)
            
            xi = torch.cat((offsets.unsqueeze(-1), xi),1)
            xi = self.encoder(xi)
 
            xi = torch.concat((xi, x_memo), -1)
 
            xi = self.conv1(xi, edge_index, edge_attr)
            # # xi = F.dropout(xi, training=self.training)
            xi = self.conv2(xi, edge_index, edge_attr)
            # # xi = F.dropout(xi, training=self.training)
            xi = self.conv3(xi, edge_index, edge_attr)
            # # xi = F.dropout(xi, training=self.training)
            xi = self.conv4(xi, edge_index, edge_attr)
            # # xi = F.dropout(xi, training=self.training)
            xi = self.conv5(xi, edge_index, edge_attr)
            #rooted graph gives diagonal entry
            decode = xi.reshape((self.batch_number, self.nodes_number, -1))
            decode = decode[:, self.bz_number*self.dimq + i, :] #shape (batch_number, hidden_channels)            
            end_embedding[i] = decode
            
            
        
    def forward2(self, x, x_memo, edge_index, edge_attr, bz_number, dimq, omega_p, batch):
        omega_p = torch.tensor(omega_p).float()
        x = x.float()
        edge_attr = edge_attr.float()
        
        if batch is not None: 
            dimq = int(dimq[0])
            bz_number = int(bz_number[0])
        else:
            batch = torch.zeros(len(x)).int()
        
        nodes_number = dimq*(2*bz_number +1)
        
        
        batch_number = int(batch[-1]) + 1
        
        # 1-D array to only store the diagonal term
        matrix = torch.zeros((batch_number, dimq)).to(device)
        
        x_memo = x_memo.view(batch_number, dimq, -1)
        
        x_memo = x_memo.repeat_interleave(2*bz_number+1, 0)
        
        x_memo = x_memo.view(len(x), -1)
        
        
        test = torch.zeros(len(x[0])).to(device)
        
        test = self.encoder(test)
        
        
        end_embedding = torch.zeros((batch_number, dimq, len(test))) 
        
        
        
        for i in range(dimq):
            
            xi = x.detach().clone()
            root_index = torch.arange(0, len(xi))%nodes_number == bz_number*dimq + i
            
            #label the root nodes.
            xi[root_index, 2] = 1             
            
            offsets = xi[root_index,0]
            offsets = offsets.repeat_interleave(nodes_number)
            
            xi = torch.cat((offsets.unsqueeze(-1), xi),1)
            
            
            xi = self.encoder(xi)
            
            xi = torch.concat((xi, x_memo), -1)
            
            xi = self.conv1(xi, edge_index, edge_attr)
            # # xi = F.dropout(xi, training=self.training)
            xi = self.conv2(xi, edge_index, edge_attr)
            # # xi = F.dropout(xi, training=self.training)
            xi = self.conv3(xi, edge_index, edge_attr)
            # # xi = F.dropout(xi, training=self.training)
            xi = self.conv4(xi, edge_index, edge_attr)
            # # xi = F.dropout(xi, training=self.training)
            xi = self.conv5(xi, edge_index, edge_attr)
            

            xi = torch.cat((offsets.unsqueeze(-1), xi),1)

            #rooted graph gives diagonal entry
            embed = xi.reshape((batch_number, nodes_number, -1))
            embed = embed[:, bz_number*dimq + i, :] #shape (batch_number, hidden_channels)


            
            result = self.decoder_r(embed)
            matrix[:, i] = result.view(-1)
            end_embedding[:, i, :] =result
            
        return matrix.squeeze(), end_embedding
        
