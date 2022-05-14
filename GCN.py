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
        
    def forward(self, x, edge_index, edge_attr, bz_number, dimq, omega_p, batch):
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
        matrix = torch.zeros((batch_number, dimq))
        
        
        
        
        for i in range(dimq):
            
            xi = x.detach().clone()
            root_index = torch.arange(0, len(xi))%nodes_number == bz_number*dimq + i
            
            #label the root nodes.
            xi[root_index, 2] = 1             
            
            offsets = xi[root_index,0]
            offsets = offsets.repeat_interleave(nodes_number)
            
            xi = torch.cat((offsets.unsqueeze(-1), xi),1)
            
            
            
            
            xi = self.encoder(xi)
            print(xi.is_cuda)
            
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
    def __init__(self, hidden_channels, num_node_features, edge_features, rounds = 2):
        super(FloquetSolver, self).__init__()
        torch.manual_seed(12345)
        
        self.encoder= Seq(Linear(4, num_node_features),
                       ReLU(),
                       Linear(num_node_features, num_node_features))
        
        self.conv1 = BetterGCNConv(num_node_features, edge_features, hidden_channels)
        self.conv2 = BetterGCNConv(num_node_features, edge_features, hidden_channels)
        self.conv3 = BetterGCNConv(num_node_features, edge_features, hidden_channels)
        self.conv4 = BetterGCNConv(num_node_features, edge_features, hidden_channels)
        self.conv5 = BetterGCNConv(num_node_features, edge_features, hidden_channels)
        
        self.decoder_ur = Seq(Linear(num_node_features, hidden_channels),
                       ReLU(),
                       Linear(hidden_channels, 1))
        
        self.decoder_r = Seq(Linear(num_node_features+1, 256),
                       ReLU(),
                       Linear(256, 64),
                       ReLU(),
                       Linear(64, 1))
        
        # self.remainder = Seq(Linear(2, 1000),
        #                ReLU(),
        #                Linear(1000, 300),
        #                ReLU(),
        #                Linear(300, 1))

        # self.lin = nn.Linear(hidden_channels, num_classes)
        # self.bn = nn.BatchNorm1d(hidden_channels, affine=False)


    def forward(self, x, edge_index, edge_attr, bz_number, dimq, omega_p, batch):
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
        
        matrix = torch.zeros((batch_number, dimq, dimq))
        
        de =  torch.zeros((batch_number, dimq))
        
        index_offset = 0
        energy_offset = 0
        
        for i in range(dimq):
            xi = torch.zeros((x.shape[0], x.shape[1]+1))
            current_batch = -1
            
            offsets = torch.zeros(batch_number)
            
            for j in range(len(xi)):
                if current_batch != batch[j]:
                    current_batch = batch[j]
                    index_offset = int(current_batch * nodes_number)
                    energy_offset = x[index_offset + dimq*bz_number +i][0]
                    offsets[current_batch] = energy_offset
                xi[j][0] = x[j][0] - energy_offset
                xi[j][1:] = x[j]
                if xi[j][0] == 0 and j % dimq == i:
                    xi[j][3] = 1
            
            xi = self.encoder(xi)
            
            xi = self.conv1(xi, edge_index, edge_attr)
            # xi = F.dropout(xi, training=self.training)
            xi = self.conv2(xi, edge_index, edge_attr)
            # xi = F.dropout(xi, training=self.training)
            xi = self.conv3(xi, edge_index, edge_attr)
            # xi = F.dropout(xi, training=self.training)
            xi = self.conv4(xi, edge_index, edge_attr)
            # xi = F.dropout(xi, training=self.training)
            xi = self.conv5(xi, edge_index, edge_attr)
            
            #unrooted graph gives off-diagonal matrix element
            encode = graph_collapse(xi,  dimq, bz_number, batch_number)
            decode = self.decoder_ur(encode) # shape (batch_number*nodes_number)
            decode = decode.view((batch_number, -1))
            matrix[:, i, :] = decode
            matrix[:, :, i] = decode
            
            #rooted graph gives diagonal entry
            decode = xi.reshape((batch_number, nodes_number, -1))
            decode = decode[:, bz_number*dimq + i, :] #shape (batch_number, hidden_channels)


            offsets = offsets.unsqueeze(-1)

            if batch_number == 1:
                omg_p = omega_p.unsqueeze(-1).unsqueeze(-1)
            else:
                omg_p = omega_p.unsqueeze(-1)
            

            
            decode = torch.cat((decode, offsets),-1)
            
            
            # decode = torch.cat((decode+offsets,omg_p),-1)
            decode = self.decoder_r(decode)
            # decode = torch.cat((decode+offsets,omg_p),-1)
            # decode = self.remainder(decode)
            
            # for n in range(len(matrix)):
                # matrix[n, i, i] = (decode[n]+offsets[n]).remainder(omg_p[n])[0]
                # de[n,i] = decode[n]
                
            matrix[:, i, i] = decode.view(-1)
            
            
        return matrix.squeeze(), de
        
            
            
        
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = self.bn(x)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = self.bn(x)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = self.bn(x)
        x = x.relu()
        x = self.conv4(x, edge_index)
        x = self.bn(x)
        x = x.relu()
        x = self.conv5(x, edge_index)
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0, training=self.training)
        x = self.lin(x)

        return x