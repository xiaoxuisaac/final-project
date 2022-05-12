# This data handling code is adapted from the PyTorch geometric collection of google colab notebooks, a fantastic resource for getting started with GNNs. https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.transforms import Constant
# import the graph classifier you built in the last step
from GCN import FloquetSolver
from dataset import FloquetDataset

import sys

# - - - DATA PREPARATIONS - - -
dataset = FloquetDataset(
    root='data/Mixed_small_ss_dim10')

print()
print(f'Dataset: {dataset}:')
print('====================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
data = dataset[0]  # Get the first graph object.
print()
print(data)
print('=============================================================')
# Gather some statistics about the first graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Contains self-loops: {data.contains_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

torch.manual_seed(12345) # for reproducibility
dataset = dataset.shuffle()

train_dataset = dataset[:4000]
test_dataset = dataset[4000:]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)

# Finally, we've got the train loader and the test loader! Time to start doing the actual training!
# "A data scientist's job is 90% data, 10% science"
# - - - TRAINING - - -

model = FloquetSolver(hidden_channels=64, num_node_features=20, edge_features=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay= 0.001)
MSELoss = torch.nn.MSELoss()

ENERGY_OFFSET = 2000
COUPLING_WEIGHT = 100
DIAG_WEIGHT = 0.01

def find_quasi(mats, evals, omega_p):
    if len(mats.shape) == 3:
        quasi = torch.zeros(mats.shape[:2])
        for i in range(len(mats)):
             quasi[i] = find_quasi(mats[i], evals[i], omega_p[i])
        return quasi
    quasi = mats.diag() #+ torch.tensor(evals).float() #add the energy difference to the original energy
    return quasi
    
    # omega_p = torch.tensor(omega_p).float()
    return quasi.sort(descending = True)[0]
    # return quasi.remainder(omega_p).sort(descending = True)[0]
    
    
    quasi = quasi + ENERGY_OFFSET
    
    Heff = mats + torch.diag(quasi - mats.diag())
    
    
    _, predict, _ = torch.linalg.svd(Heff)
    predict = predict - ENERGY_OFFSET
    predict,_ = predict.sort(descending = True)
    
    return predict
    
def criterion(mats, y, evals, omega_p):   
    predict = find_quasi(mats, evals, omega_p).view(-1)
    
    for i in range(len(y)):
        if y[i] == -1964.:
            y[i] = predict[i]
    
    return MSELoss(predict, y)
    

# for data in train_loader: 
    # break

# m, de = model(data.x, data.edge_index, data.edge_attr, data.bz_number, data.dimq, 
            # data.omega_p, data.batch)
# criterion(m, data.y, data.evals, data.omega_p)


def offset_loss(offset):
    offset = offset.view(-1).abs()
    l = torch.nn.ReLU()(torch.exp((offset-0.4))-1)
    
    return l.mean()*0.4
    

def train():
    model.train()
    counter = 0
    total_loss = 0
    for data in train_loader:  # Iterate in batches over the training dataset.
        out, de = model(data.x, data.edge_index, data.edge_attr, 
                    data.bz_number, data.dimq,  data.omega_p, data.batch) # Perform a single forward pass.
        
        loss0 = criterion(out, data.y.float(), data.evals, data.omega_p)  # Compute the loss.
                
        out_diag = torch.diagonal(out,dim1=-2, dim2=-1)
        
        loss_diag = (out_diag*out_diag).sum()        
        loss_matrix = (out*out).sum()
        
        loss_diag_norm = loss_diag/len(out.reshape(-1)) * DIAG_WEIGHT
        
        loss_couple_norm = (loss_matrix-loss_diag)/len(out.reshape(-1))*COUPLING_WEIGHT
        
        
        loss = loss0 
        
        
        if data.batch is None:
            batch_number = 1
        else:
            batch_number = data.batch[-1]+1
            
        total_loss += loss.detach().numpy()*batch_number.detach().numpy()
        
        
        print(counter, loss.detach().numpy(), loss0.detach().numpy(),
              loss_couple_norm.detach().numpy())
        counter += 1
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        
    return total_loss/ len(train_loader.dataset)
        
def test(loader):
    model.eval()
    total_loss = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out, de = model(data.x, data.edge_index, data.edge_attr, 
                    data.bz_number, data.dimq, data.omega_p, data.batch)
        
        if data.batch is None:
            batch_number = 1
        else:
            batch_number = data.batch[-1]+1
        
        total_loss += criterion(out, data.y, data.evals, data.omega_p) * batch_number
    
    return total_loss / len(loader.dataset)

train_accs = []
test_accs = []

for epoch in range(1, 2000):
    print(epoch)
    train_acc = train()
    if epoch % 1 == 0:
        # train_acc = test(train_loader)
        test_acc = test(test_loader)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        train_accs.append(train_acc)
        test_accs.append(test_acc)
# torch.save(model.state_dict(),"gcn_test1.pt")













