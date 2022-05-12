# This data handling code is adapted from the PyTorch geometric collection of google colab notebooks, a fantastic resource for getting started with GNNs. https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Constant
# import the graph classifier you built in the last step
from GCN import FloquetSolver
from dataset import FloquetDataset
import numpy as np
import sys, os, random

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--lr", type=float, default=0.002)
parser.add_argument("--weight_decay", type=float, default=0.001)
parser.add_argument("--hidden_channel", type=int, default=64)
parser.add_argument("--node_features", type=int, default=20)
parser.add_argument("--epoches", type=int, default=100)

args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

# - - - DATA PREPARATIONS - - -
dataset = FloquetDataset(
    root=os.path.abspath( os.path.dirname( __file__ ) )+'/data/Mixed_small_ss_dim10')

data = dataset[0]  # Get the first graph object.


torch.manual_seed(12345) # for reproducibility
dataset = dataset.shuffle()

train_dataset = dataset[:4000]
test_dataset = dataset[4000:]


train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)



# Finally, we've got the train loader and the test loader! Time to start doing the actual training!
# "A data scientist's job is 90% data, 10% science"
# - - - TRAINING - - -

model = FloquetSolver(hidden_channels=args.hidden_channel,
                      num_node_features=args.node_features,edge_features=3)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay= args.weight_decay)
MSELoss = torch.nn.MSELoss()

ENERGY_OFFSET = 2000
COUPLING_WEIGHT = 100
DIAG_WEIGHT = 0.01

    
def criterion(mats, y, evals, omega_p):   
    predict = mats.view(-1)
    
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
        out = model(data.x, data.edge_index, data.edge_attr, 
                    data.bz_number, data.dimq,  data.omega_p, data.batch) # Perform a single forward pass.
        
        loss0 = criterion(out, data.y.float(), data.evals, data.omega_p)  # Compute the loss.
                        
        
        loss = loss0 
        
        
        if data.batch is None:
            batch_number = 1
        else:
            batch_number = data.batch[-1]+1
            
        total_loss += loss.detach().numpy()*batch_number.detach().numpy()
        
        
        print(counter, loss.detach().numpy())
        counter += 1
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        
    return total_loss/ len(train_loader.dataset)
        
def test(loader):
    model.eval()
    total_loss = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.edge_attr, 
                    data.bz_number, data.dimq, data.omega_p, data.batch)
        
        if data.batch is None:
            batch_number = 1
        else:
            batch_number = data.batch[-1]+1
        
        total_loss += criterion(out, data.y, data.evals, data.omega_p) * batch_number
    
    return total_loss.detach().numpy() / len(loader.dataset)

def main():
    
    name = random.randint(1000,9999)
    train_accs = []
    test_accs = []
    
    with open(f'info{name:04d}.txt', 'w+') as f:
        f.write(str(args))
    
    for epoch in range(args.epoches):
        print(epoch)
        train_acc = train()
        if epoch % 1 == 0:
            # train_acc = test(train_loader)
            test_acc = test(test_loader)
            print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            np.array(train_accs).tofile(f'train_accs{name:04d}.py')
            np.array(test_accs).tofile(f'test_accs{name:04d}.py')
            torch.save(model.state_dict(),f"gcn_test1{name:04d}.pt")


if __name__ == "__main__":
    main()










