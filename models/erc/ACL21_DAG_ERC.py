import torch, fitlog, os, random
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import numpy as np, itertools, random, copy, math
from transformers import AdamW
from .DAG_ERC_utils import *

def random_parameters(args):
    # 随机参数
    params = {
        'lr': [5e-6, 8e-6, 1e-5], # 5e-5效果就不好了
        'bz': [2, 4, 8], 
        'dp': [0.1, 0.3, 0.5], 
    }
    args.seed = random.randint(0,10000)
    args.learning_rate = random.choice(params['lr'])
    args.drop_rate = random.choice(params['dp'])
    args.batch_size = int(random.choice(params['bz']))
    
    return args

def config_for_model(args, scale='base'):
    ## 补充模型参数
    args.model_base = 'roberta'
    args.special_data_path = f'{args.basic_data_path}.{args.model_base}'
    args.plm = None # 无需分词，utt表示已给定
    args.gnn_layers = 2

    ## 数据集来源不同
    args.data_file = {
        'train': f'{args.dir_data}/{args.task_name}/train_data_with_roberta_feature.json',
        'valid': f'{args.dir_data}/{args.task_name}/dev_data_with_roberta_feature.json',
        'test' : f'{args.dir_data}/{args.task_name}/test_data_with_roberta_feature.json',
    }

    ## 随机化训练参数
    if 'search' in args.method:
        args = random_parameters(args)

    return args

def data_emdedding(args):
    args.logger.info("embedding dataset, waiting ... ") 
    
    # 数据源不同，重新导入数据
    from utils.dataloader_erc import ERCDataset
    dataset = ERCDataset(args, other='cls')
    
    # 对数据进行数据化处理
    for desc, data in dataset.datas.items():
        data_embed = []
        for item in data:
            # pre_train_input = item['utterances'] # 仅使用utterance文本
            speakers = torch.tensor([dataset.speakers['stoi'][speaker] for speaker in item['speakers']])
            labels = torch.tensor([dataset.labels['stoi'][label] for label in item['labels']])
            utt_cls = torch.tensor(item['cls'])
            item_embed = {
                'index': item['index'],
                'speakers': speakers,
                'labels': labels,
                'cls': utt_cls,
            }
            data_embed.append(item_embed)
        dataset.datas_embed[desc] = data_embed

    return dataset

def get_special_dataset(args):
    args.logger.info("loading dataset, waiting ... ")
    if os.path.exists(args.special_data_path):
        dataset = torch.load(args.special_data_path)
    else:
        from utils.dataloader_erc import get_basic_dataset
        dataset = data_emdedding(args) # 重新导入数据
        torch.save(dataset, args.special_data_path)

    ## dataloader
    shuffle = {'train': True, 'valid': False, 'test': False}
    for desc, data_embed in dataset.datas_embed.items():
        dataset.datas_loader[desc] = DataLoader(dataset=data_embed, batch_size=args.batch_size, shuffle=shuffle[desc], collate_fn=dataset.collate_fn)

    ## 定义模型输入内容
    dataset.batch_cols = {
        'speakers': -1, 
        'labels': -1, 
        'cls': 0,
    }
    args.dataset = dataset

    return args

def import_model(args):
    args = config_for_model(args) # 添加模型参数, 获取任务数据集
    fitlog.set_rng_seed(args.seed)
    args = get_special_dataset(args) 
    model = DAGERC_fushion(args=args) # 这个才行   

    return model

class DAGERC_fushion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dataset = args.dataset

        self.emb_dim = 1024
        self.hidden_dim = 300
        self.linear = nn.Sequential(
            nn.Linear(self.emb_dim, self.hidden_dim),
            # nn.Dropout(0.3),
            nn.ReLU(),
            # nn.Dropout(0.3),
        ) # 特征降维
        self.dropout = nn.Dropout(args.drop_rate)

        

        gats = []
        for _ in range(args.gnn_layers):
            # gats += [GAT_dialoggcn(args.hidden_dim)]
            gats += [GAT_dialoggcn_v1(self.hidden_dim)]
        self.gather = nn.ModuleList(gats)

        grus_c = []
        for _ in range(args.gnn_layers):
            grus_c += [nn.GRUCell(self.hidden_dim, self.hidden_dim)]
        self.grus_c = nn.ModuleList(grus_c)

        grus_p = []
        for _ in range(args.gnn_layers):
            grus_p += [nn.GRUCell(self.hidden_dim, self.hidden_dim)]
        self.grus_p = nn.ModuleList(grus_p)
        
        fcs = []
        for _ in range(args.gnn_layers):
            fcs += [nn.Linear(self.hidden_dim * 2, self.hidden_dim)]
        self.fcs = nn.ModuleList(fcs)

        self.nodal_att_type = None
        
        in_dim = self.hidden_dim * (args.gnn_layers + 1) + self.emb_dim

        # output mlp layers
        self.mlp_layers = 2
        layers = [nn.Linear(in_dim, self.hidden_dim), nn.ReLU()]
        for _ in range(self.mlp_layers - 1):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()]
        layers += [self.dropout]
        layers += [nn.Linear(self.hidden_dim, args.dataset.n_class)]
        self.out_mlp = nn.Sequential(*layers)

        self.attentive_node_features = attentive_node_features(in_dim)

        self.loss_ce = CrossEntropyLoss(ignore_index=-1)


    def _optimizer(self, lr=None):
        optimizer = AdamW(self.parameters() , lr=self.args.learning_rate)
        scheduler = None
        return optimizer, scheduler

    def get_adj_mask(self, speakers):
        bz, mun = speakers.shape
        ## utt邻接矩阵 / speakers mask
        utt_adjs = torch.zeros(bz, mun, mun).type_as(speakers)
        speakers_mask = torch.zeros(bz, mun, mun).type_as(speakers)
        for ai, (adj, mask) in enumerate(zip(utt_adjs, speakers_mask)):
            speaker = speakers[ai]
            # 1) utt 邻接矩阵
            for si, s in enumerate(speaker):
                if s == -1: break # 后面已无实际 utt
                for k in range(si-1, -1, -1): # 往回追溯，
                    adj[si][k] = 1 # 非同一说话者的utt均指向当前utt
                    if speaker[k] == s: break # 追溯到相同说话者的utt了 
            # 2）speaker mask
            speaker_fact_num = torch.sum(speaker >= 0).item()
            for i in range(speaker_fact_num):
                for j in range(speaker_fact_num):
                    if speaker[i] == speaker[j]: mask[i][j] = 1

        return utt_adjs, speakers_mask

    def forward(self, batch, stage='train'):
        '''
        adj: 当前utt与上一个utt(同speaker)之间的utts, 都指向当前utt (有向)
        s_mask: 每个speaker指向自己, 包括自身 (无向)
        '''
        speakers, labels, features = batch['speakers'], batch['labels'], batch['cls']
        adj, s_mask = self.get_adj_mask(speakers) # utt 连接图
        H0 = self.linear(features) # utt特征降维 1021 -> 300

        utt_mask = labels >= 0
        lengths = torch.sum(utt_mask, dim=-1)
        num_utter = features.size()[1]
        H = [H0] 
        for l in range(self.args.gnn_layers):
            C = self.grus_c[l](H[l][:,0,:]).unsqueeze(1) # 首个utt, 经过一层GRU  -> seq_len=1
            M = torch.zeros_like(C).squeeze(1) 
            # P = M.unsqueeze(1) 
            P = self.grus_p[l](M, H[l][:,0,:]).unsqueeze(1)  
            #H1 = F.relu(self.fcs[l](torch.cat((C,P) , dim = 2)))
            #H1 = F.relu(C+P)
            H1 = C + P
            for i in range(1, num_utter):
                # print(i,num_utter)
                _, M = self.gather[l](H[l][:,i,:], H1, H1, adj[:,i,:i], s_mask[:,i,:i]) # 公式(5)
                C = self.grus_c[l](H[l][:,i,:], M).unsqueeze(1) # 公式(6)
                P = self.grus_p[l](M, H[l][:,i,:]).unsqueeze(1) # 公式(7)
                # P = M.unsqueeze(1)
                #H_temp = F.relu(self.fcs[l](torch.cat((C,P) , dim = 2)))  
                #H_temp = F.relu(C+P)
                H_temp = C+P # 公式(8)
                H1 = torch.cat((H1 , H_temp), dim = 1)  
                # print('H1', H1.size())
                # print('----------------------------------------------------')
            H.append(H1)
        H.append(features) # 公式（9）
        
        H = torch.cat(H, dim = 2) 

        H = self.attentive_node_features(H, lengths, self.nodal_att_type) 

        logits = self.out_mlp(H)
        loss = self.loss_ce(logits[utt_mask, :], labels[utt_mask])

        return {
            'loss': loss,
            'logits': logits,
            'preds': torch.argmax(logits[utt_mask,:], dim=-1).cpu(),
            'labels': labels[utt_mask],
        }
