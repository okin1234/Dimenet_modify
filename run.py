from Dimenet import layers
from Dimenet import dimenet
from Dimenet import dataset
from Dimenet.dataset import make_datasets
from Dimenet.dimenet import DimeNet
from Dimenet import utils
from Dimenet.utils import EarlyStopping

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
from transformers import get_cosine_schedule_with_warmup
import time


class trainer():
    def __init__(self, base_dir, hidden_channels=128, out_channels=1, num_blocks=4, num_bilinear=64, basis_emb_size=8, out_emb_channels=256, num_spherical=7, num_radial=6, cutoff=5.0, max_num_neighbors=32, envelope_exponent=5, num_before_skip=1, num_after_skip=2, num_output_layers=3, act='swish', output_init='GlorotOrthogonal', random_state=1000, num_state=1,
                num_xtb_data=None, num_g09_data=None, load=False, g09_data_cut=None, with_rdkit=False, rdkit_rank=10, to_class=False, class_num=5, cut_mode='qcut', onehot=False, xtb_state=0, g09_state=1):
        self.base_dir = base_dir
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_bilinear = num_bilinear
        self.basis_emb_size = basis_emb_size
        self.out_emb_channels = out_emb_channels
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.envelope_exponent = envelope_exponent
        self.num_before_skip = num_before_skip
        self.num_after_skip = num_after_skip
        self.num_output_layers = num_output_layers
        self.act = act
        self.output_init = output_init
        self.random_state = random_state
        
        self.num_state = num_state
        self.num_xtb_data = num_xtb_data
        self.num_g09_data = num_g09_data
        self.load = False
        self.g09_data_cut = g09_data_cut
        self.with_rdkit = with_rdkit
        self.rdkit_rank = rdkit_rank
        self.to_class = to_class
        self.class_num = class_num
        self.cut_mode = cut_mode
        self.onehot = onehot
        self.xtb_state = xtb_state
        self.g09_state = g09_state

        
    def make_train_val_set(self):
        dataset_maker = make_datasets(self.base_dir)
        train_graphs, train_targets, val_graphs, val_targets = dataset_maker.make_graph_datas(self.num_xtb_data, self.num_g09_data, load=self.load, random_state=self.random_state, g09_data_cut=self.g09_data_cut, with_rdkit=self.with_rdkit, rdkit_rank=self.rdkit_rank, to_class=self.to_class, class_num=self.class_num, cut_mode=self.cut_mode, onehot=self.onehot, xtb_state=self.xtb_state, g09_state=self.g09_state)

        return train_graphs, train_targets, val_graphs, val_targets


    def train(self, batch_size=16, lr=0.001, epochs=100, model_dict=None, 
             wandb_on=True, wandb_project=None, wandb_name=None, wandb_group=None, wandb_memo=None
            ):
        if wandb_on:
            import wandb
            model_dict['batch_size'] = batch_size
            model_dict['lr'] = lr
            model_dict['epochs'] = epochs
            
            wandb.init(project=wandb_project, reinit=True, group=wandb_group, notes=wandb_memo, config=model_dict)
            wandb.run.name = wandb_name
            wandb.run.save()
        
        if wandb_name is None:
            wandb_name = 'testing'

        self.set_seed(self.random_state)

        if self.act == 'swish':
            act = swish

        train_graphs, train_targets, val_graphs, val_targets = self.make_train_val_set()

        train_dataset = self.data_to_pyg_form(train_graphs, train_targets)
        val_dataset = self.data_to_pyg_form(val_graphs, val_targets)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=self.random_state)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DimeNet(hidden_channels=self.hidden_channels, out_channels=self.out_channels ,
                         num_blocks=self.num_blocks, num_bilinear=self.num_bilinear, basis_emb_size=self.basis_emb_size, 
                         out_emb_channels=self.out_emb_channels, num_spherical=self.num_spherical,
                         num_radial=self.num_radial, cutoff=self.cutoff, max_num_neighbors=self.max_num_neighbors,
                         envelope_exponent=self.envelope_exponent, num_before_skip=self.num_before_skip,
                         num_after_skip=self.num_after_skip, num_output_layers=self.num_output_layers,
                         act=act, output_init=self.output_init)
        optim = Adam(model.parameters(), lr=lr, amsgrad=True)
        model = model.to(device)
        #scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps=100, num_training_steps=(len(train_loader)*epochs))
        scheduler = ExponentialLR(optim, gamma=0.96)
        scheduler_warmup = GradualWarmupScheduler(optim, multiplier=1.0, total_epoch=1, after_scheduler=scheduler)
        early_stopping = EarlyStopping(patience = 5, verbose = True)

        best_epoch = None
        best_val_loss = None

        for epoch in range(epochs):
            loss_all = 0
            step = 0
            start_time = time.time()
            model.train()

            for batch in train_loader:
                batch = batch.to(device)
                optim.zero_grad()
                output = model(batch)

                loss = F.l1_loss(output.squeeze(), batch.y)
                loss_all += loss.item() * batch.num_graphs
                loss.backward()
                optim.step()
                #scheduler.step()

            train_loss = loss_all / len(train_loader.dataset)
            val_loss = self.model_eval(model, device, val_loader)
            end_time = time.time()
            running_time = round((end_time - start_time)/60, 2)
            
            if best_val_loss is None or val_loss <= best_val_loss:
                best_epoch = epoch+1
                best_val_loss = val_loss
                torch.save(model.state_dict(), "./save_model/{0}_{1}_{2}.pt".format(wandb_name, epoch, val_loss))
            print('Epoch: {:03d}, Train MAE: {:.7f}, Validation MAE: {:.7f}, lr: {:.7f}, time: {:.7f}'.format(epoch+1, train_loss, val_loss, optim.param_groups[0]['lr'], running_time))
            
            if wandb_on:
                wandb.log({'Train mae' : train_loss, 'Val mae' : val_loss, 'lr' : optim.param_groups[0]['lr'], 'time' : running_time})

            scheduler_warmup.step()
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print('===================================================================================')
        print('Best Epoch:', best_epoch)
        print('Best Val MAE:', best_val_loss)
        if wandb_on:
            wandb.finish()


    def eval(self, selected_model=None, batch_size=32):

        self.set_seed(self.random_state)

        if self.act == 'swish':
            act = swish

        train_graphs, train_targets, val_graphs, val_targets = self.make_train_val_set()

        train_dataset = self.data_to_pyg_form(train_graphs, train_targets)
        val_dataset = self.data_to_pyg_form(val_graphs, val_targets)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=self.random_state)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DimeNet(hidden_channels=self.hidden_channels, out_channels=self.out_channels,
                         num_blocks=self.num_blocks, num_bilinear=self.num_bilinear, basis_emb_size=self.basis_emb_size, 
                         out_emb_channels=self.out_emb_channels, num_spherical=self.num_spherical,
                         num_radial=self.num_radial, cutoff=self.cutoff, max_num_neighbors=self.max_num_neighbors,
                         envelope_exponent=self.envelope_exponent, num_before_skip=self.num_before_skip,
                         num_after_skip=self.num_after_skip, num_output_layers=self.num_output_layers,
                         act=act, output_init=self.output_init)
        model.load_state_dict(selected_model)

        train_loss, train_pred, train_target = self.model_eval(model, device, train_loader)
        val_loss, val_pred, val_target = self.model_eval(model, device, val_loader)


        print('===================================================================================')

        return train_loss, train_pred, train_target, val_loss, val_pred, val_target

    def data_to_pyg_form(self, graphs, targets):
        data_list=[]
        for n, graph in enumerate(graphs):
            target=torch.tensor(targets[n])
            x=torch.tensor(graph['atom'])
            pos=torch.tensor(graph['pos'])
            state=torch.tensor(graph['state'])
            data = Data(x=x, pos=pos, state=state, y=target)
            data_list.append(data)
        return data_list

    def set_seed(self, seed):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def model_eval(self, model, device, loader):
        model.eval()
        error = 0
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                output = model(data)
                error += (output.squeeze() - data.y).abs().sum().item()

        return error / len(loader.dataset)
