#from Dimenet import *
#from layers import *
#from dimenet import *
#from dataset import *
from Dimenet import layers
from Dimenet import dimenet
from Dimenet import dataset
from Dimenet.dimenet import DimeNet
from Dimenet.dataset import make_graph_datas

import torch
#from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch_geometric.nn.acts import swish
from torch_geometric.data import Data, DataLoader
from warmup_scheduler import GradualWarmupScheduler
import numpy as np
import random

def data_to_pyg_form(graphs, targets):
    data_list=[]
    for n, graph in enumerate(graphs):
        target=torch.tensor(targets[n])
        x=torch.tensor(graph['atom'])
        pos=torch.tensor(graph['pos'])
        state=torch.tensor(graph['state'])
        data = Data(x=x, pos=pos, state=state, y=target)
        data_list.append(data)
    return data_list

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def model_eval(model, device, loader):
    model.eval()
    error = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            error += (output.squeeze() - data.y).abs().sum().item()

    return error / len(loader.dataset)

def main(hidden_channels=128, out_channels=1 , num_blocks=4, num_bilinear=64, basis_emb_size=8, out_emb_channels=256, num_spherical=7, num_radial=6, cutoff=5.0, max_num_neighbors=32, envelope_exponent=5, num_before_skip=1, num_after_skip=2, num_output_layers=3, act='swish', output_init='GlorotOrthogonal', random_state=1000, batch_size=32, lr=0.001, epochs=100, 
         num_xtb_data=None, num_g09_data=None, load=False, g09_data_cut=None, with_rdkit=False, rdkit_rank=10, to_class=False, class_num=5, cut_mode='qcut', onehot=False
        ):

    set_seed(random_state)

    if act == 'swish':
        act = swish
    
    train_graphs, train_targets, val_graphs, val_targets = make_graph_datas(num_xtb_data, num_g09_data, load=load, random_state=random_state, g09_data_cut=g09_data_cut, with_rdkit=with_rdkit, rdkit_rank=rdkit_rank, to_class=to_class, class_num=class_num, cut_mode=cut_mode, onehot=onehot)

    train_dataset = data_to_pyg_form(train_graphs, train_targets)
    val_dataset = data_to_pyg_form(val_graphs, val_targets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=random_state)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DimeNet(hidden_channels=hidden_channels, out_channels=out_channels ,
                     num_blocks=num_blocks, num_bilinear=num_bilinear, basis_emb_size=basis_emb_size, 
                     out_emb_channels=out_emb_channels, num_spherical=num_spherical,
                     num_radial=num_radial, cutoff=cutoff, max_num_neighbors=max_num_neighbors,
                     envelope_exponent=envelope_exponent, num_before_skip=num_before_skip,
                     num_after_skip=num_after_skip, num_output_layers=num_output_layers,
                     act=swish, output_init=output_init)
    optim = Adam(model.parameters(), lr=lr, amsgrad=True)
    model = model.to(device)
    scheduler = ExponentialLR(optim, gamma=0.99)
    scheduler_warmup = GradualWarmupScheduler(optim, multiplier=1.0, total_epoch=4, after_scheduler=scheduler)

    best_epoch = None
    best_val_loss = None

    for epoch in range(epochs):
        loss_all = 0
        step = 0
        model.train()

        for batch in train_loader:
            batch = batch.to(device)
            optim.zero_grad()
            output = model(batch)

            loss = F.l1_loss(output.squeeze(), batch.y)
            loss_all += loss.item() * batch.num_graphs
            loss.backward()
            optim.step()
            
        train_loss = loss_all / len(train_loader.dataset)
        val_loss = model_eval(model, device, val_loader)

        if best_val_loss is None or val_loss <= best_val_loss:
            best_epoch = epoch+1
            best_val_loss = val_loss
            #torch.save(model.state_dict(), "./save_model/{0}_{1}_{2}.pt".format(wandb_name, epoch, val_loss))
        print('Epoch: {:03d}, Train MAE: {:.7f}, Validation MAE: {:.7f}, lr: {:.7f}'.format(epoch+1, train_loss, val_loss, optim.param_groups[0]['lr']))
        #wandb.log({'Train mae' : train_loss, 'Val mae' : val_loss, 'lr' : optim.param_groups[0]['lr']})
        
        scheduler_warmup.step()

    print('===================================================================================')
    print('Best Epoch:', best_epoch)
    print('Best Val MAE:', best_val_loss)