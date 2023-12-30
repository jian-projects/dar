import torch, os, fitlog, random, copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, AdamW
from transformers import get_cosine_schedule_with_warmup
from config import random_parameters
from utils.dataloader_erc import *
"""
| dataset     | meld  | iec   | emn   | ddg   |

| baseline    | 67.25 | 69.74 | 40.94 | 00.00 |
| performance | 00.00 | 00.00 | 00.00 | 00.00 |

"""
baselines = {
    'base': {'meld': 0.645, 'iec': 0., 'emn': 0., 'ddg': 0}, 
    'large': {'meld': 0.000, 'iec': 0., 'emn': 0., 'ddg': 0},
}

# 重写数据构造方式
class ERCDataset_Pro(ERCDataset_Multi):
    def __init__(self, data_path, lower=False):
        self.lower = lower
        self.name = ['erc', data_path.split('/')[-2]]
        self.context_num = 8
        self.max_seq_len = 256
        self.data_path = data_path 
        self.container_init() # 初始化容器信息
        for desc in ['train', 'valid', 'test']:
            self.datas['text'][desc] = self.get_dataset(data_path, desc)
        self.n_class = len(self.tokenizer_['labels']['ltoi']) 

    def container_init(self, only='all'):
        self.info = {
            'dialog_num': {'train': 0, 'valid': 0, 'test': 0}, # 数据对话数
            'max_utt_num': {'train': 0, 'valid': 0, 'test': 0}, # dialog最大utterance数
            'max_token_num': {'train': 0, 'valid': 0, 'test': 0}, # utterance最大token数
            'total_samples_num': {'train': 0, 'valid': 0, 'test': 0}, # 重构样本数量
            'emotion_category': {}, # 情绪类别
        }
        # 初始化数据集要保存的内容 
        path_tokenizer_ = self.data_path + 'tokenizer_'
        if os.path.exists(path_tokenizer_):
            self.tokenizer_ = torch.load(path_tokenizer_)
            self.path_tokenizer_ = None
        else:
            self.tokenizer_ = {
                'labels': { 'ltoi': {}, 'itol': {}, 'count': {}, 'class': [] }, # 分类标签字典
                'speakers': { 'stoi': {}, 'itos': {}, 'count': {}}, # speaker字典
            }
            self.path_tokenizer_ = path_tokenizer_


        self.datas_ori = {
            'text': {},        # 解析后纯文本
            'vector': {},   # 文本分词后数字化表示
            'dataloader': {},  # dataloader用于构建batch
        }
        self.datas = {
            'text': {},        # 解析后纯文本
            'vector': {},   # 文本分词后数字化表示
            'dataloader': {},  # dataloader用于构建batch
        }

        self.tokens_add = [] # speaker 匿名token

    def add_prompt(self, utterance, speaker, contexts, add_c=1):
        context_num = self.context_num
        eos_token, mask_token = self.tokenizer.eos_token, self.tokenizer.mask_token
        speaker_text = speaker + ': ' + utterance
        if add_c: contexts.append(speaker_text) # 记录在 context 中

        query_prompt = f'For utterance: {speaker_text} {eos_token} {speaker} feels {mask_token}'
        sample_context = f' {eos_token} '.join(contexts[-context_num:]) # 上下文，保留前8句
        final_text = sample_context + f' {eos_token} ' + query_prompt
        
        return {
            'text': final_text, 
            'speaker': speaker,
        }, contexts

    def prompt(self, dialogs, method='ori'):
        samples, samples_ = [], []
        for dialog in dialogs:
            contexts = []
            for ui in range(len(dialog['texts'])):
                sample, contexts = self.add_prompt(
                    utterance=dialog['texts'][ui], 
                    speaker=dialog['speakers'][ui], 
                    contexts=contexts
                )
                sample['index'], sample['label'] = len(samples), dialog['labels'][ui]
                samples.append(sample)

                ## 随机添加负样本，随机从context中选择一个utterance构造prompt
                if ui>3 and random.random()<0.2:
                    select_ui = random.choice(list(range(max(0,ui-8),ui)))
                    sample, _ = self.add_prompt(
                        utterance=dialog['texts'][select_ui], 
                        speaker=dialog['speakers'][select_ui], 
                        contexts=contexts,
                        add_c=0, # 不记录 contexts
                    )
                    sample['index'], sample['label'] = len(samples_), dialog['labels'][select_ui]
                    samples_.append(sample)
        if method == 'ori': return samples
        if method == 'add': return samples_

    def add_trainset(self, truncate='first'):
        tokenizer = self.tokenizer
        speaker_fn, label_fn = self.tokenizer_['speakers']['stoi'], self.tokenizer_['labels']['ltoi']
        dialogs = self.datas['text']['train']
        samples_ = self.prompt(dialogs, method='add')
        samples_embed = self.datas_ori['vector']['train']
        for sample in samples_:
            embedding = tokenizer(sample['text'], return_tensors='pt')
            input_ids, attention_mask = self.vector_truncate(embedding, truncate=truncate)
            speaker, label = speaker_fn[sample['speaker']], label_fn[sample['label']]
            sample_embed = {
                'index': len(samples_embed),
                'input_ids': input_ids.squeeze(dim=0),
                'attention_mask': attention_mask.squeeze(dim=0),
                'speaker': torch.tensor(speaker),
                'label': torch.tensor(label),
            }
            samples_embed.append(sample_embed)  

        self.datas['vector']['train'] = samples_embed

    def get_vector(self, tokenizer, truncate='tail', only=None):
        self.tokenizer = tokenizer
        speaker_fn, label_fn = self.tokenizer_['speakers']['stoi'], self.tokenizer_['labels']['ltoi']
        for desc, dialogs in self.datas['text'].items():
            if only is not None and desc!=only: continue
            samples, samples_embed = self.prompt(dialogs, method='ori'), [] # 重构样本, 增加prompt
            for sample in samples:
                embedding = tokenizer(sample['text'], return_tensors='pt')
                input_ids, attention_mask = self.vector_truncate(embedding, truncate=truncate)
                speaker, label = speaker_fn[sample['speaker']], label_fn[sample['label']]
                sample_embed = {
                    'index': sample['index'],
                    'input_ids': input_ids.squeeze(dim=0),
                    'attention_mask': attention_mask.squeeze(dim=0),
                    'speaker': torch.tensor(speaker),
                    'label': torch.tensor(label),
                }
                samples_embed.append(sample_embed)

            self.datas['vector'][desc] = samples_embed
            self.datas_ori['vector'][desc] = samples_embed # 备份基本数据集

    def collate_fn(self, samples):
        ## 获取 batch
        inputs = {}
        for col, pad in self.batch_cols.items():
            if 'ids' in col or 'mask' in col:  
                inputs[col] = pad_sequence([sample[col] for sample in samples], batch_first=True, padding_value=pad)
            else: 
                inputs[col] = torch.tensor([sample[col] for sample in samples])

        return inputs


def config_for_model(args, scale='base'):
    scale = args.params_model.scale
    args.params_model.about = 'spcl'
    args.params_model.plm = f'princeton-nlp/sup-simcse-roberta-{scale}'
    
    args.params_model.data = args.params_file.data + f'dataset.{args.params_model.name[0]}.' + scale
    args.params_model.baseline = baselines[scale][args.task[-1]]

    args.params_model.tokenizer = None
    args.params_model.optim_sched = ['AdamW_', 'cosine']

    args.params_model.use_cl  = 0  # 是否使用课程学习
    args.params_model.use_scl = 0 # 是否使用对比学习
    args.params_model.cxt_num = 8 # 保留前八句作为上下文
    
    return args
                      
def import_model(args):
    ## 1. 更新参数
    args = config_for_model(args) # 添加模型参数, 获取任务数据集
    args = random_parameters(args)
    fitlog.set_rng_seed(args.params_train.seed)
    
    ## 2. 导入数据
    dataset = get_special_dataset(
        info_data=[args.params_model.data, args.params_file.data],
        info_tokenizer=[args.params_model.plm, args.params_model.tokenizer],
        data_fn=ERCDataset_Pro,
        truncate='first',
    ) # for dataloader_absa
    dataset.batch_cols = {
        'index': -1,
        'input_ids': dataset.tokenizer.pad_token_id, 
        'attention_mask': 0, 
        'label': -1, 
    }
    dataset.shuffle = {'train': True, 'valid': False, 'test': False}
    args.dataset = dataset

    model = SPCL(
        args=args,
        n_class=dataset.n_class,
        plm=args.params_model.plm,
    )

    return model


class SPCL(nn.Module):
    def __init__(self, args, n_class, plm=None):
        super(SPCL, self).__init__()
        self.n_class = n_class
        self.params = args.params_model
        self.mask_token_id = args.dataset.tokenizer.mask_token_id

        self.plm_model = AutoModel.from_pretrained(plm, local_files_only=False)
        self.hidden_dim = self.plm_model.embeddings.word_embeddings.embedding_dim
        self.linear = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.clasifier = nn.Linear(self.hidden_dim, n_class)
        self.loss_ce = CrossEntropyLoss(ignore_index=-1)

        self.cluster = Cluster(args, n_class, self.encode)

    def epoch_deal(self, epoch, dataset, device):
        ## 增加训练样本
        dataset.add_trainset()

        ## 更新样本表示, 计算聚类中心
        if self.params.use_scl:
            dataset = self.cluster(epoch, dataset)

        return dataset

    def encode(self, inputs):
        # 获得一个batch中样本表示
        out_plm = self.plm_model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            output_hidden_states=True,
            return_dict=True
        )
        hidden_state = out_plm.last_hidden_state
        # 取出每个样本 <mask> 位置的表示
        mask_positions = (inputs['input_ids']==self.mask_token_id).max(dim=1)[1]
        mask_representations = hidden_state[torch.arange(len(mask_positions)), mask_positions, :]
        # 经过线性层后即可得到样本表示向量
        features = self.linear(mask_representations)

        return features

    def forward(self, inputs, stage='train'):
        features, labels = self.encode(inputs), inputs['label'] # 计算样本表示

        ## 计算交叉熵损失
        logits = self.clasifier(features)
        preds = torch.argmax(logits, dim=-1).cpu()
        loss = self.loss_ce(logits, labels)

        ## 计算对比损失
        if self.params.use_scl:
            ## 按相似性判断所属类
            centers = torch.stack([self.cluster.centers[cla] for cla in range(self.n_class)]).type_as(features)
            logits_cluster = self.cluster.similarity(features, centers)
            preds_cluster = torch.argmax(logits_cluster, dim=-1)  

            preds = preds_cluster   
            loss += self.cluster.scl_loss(features, inputs['label'], preds)

        return {
            'loss': loss,
            'labels': labels,
            'logits': logits,
            'preds': preds,
            #'logits_cluster': logits_cluster,
            #'preds_cluster': preds_cluster,
        }


class Cluster(nn.Module):
    def __init__(self, args, n_class, encode) -> None:
        super().__init__()
        self.args = args
        self.n_class = n_class
        self.device = args.params_train.device
        self.encode = encode

        self.pools, self.pool_size = {}, 64 # 样本类池
        for lab in range(n_class): self.pools[lab] = []

        self.pool_size_k = 16
        self.temp = 0.05 
        self.eps = 1e-8
        
    def gen_representation(self, dataset):
        # 清空容器
        samples_len = len(dataset.datas['vector']['train'])
        self.train_resps, self.train_labels = np.array([None]*samples_len), np.array([None]*samples_len)
        dataloader = DataLoader(
            dataset=dataset.datas['vector']['train'], 
            batch_size=128, #512, 
            shuffle=False, 
            collate_fn=dataset.collate_fn
        )

        #print('updating the instance representations, waiting ... ')
        #process_tqdm = tqdm(total=len(dataloader), position=1)
        #process_tqdm.set_description("generating ...")
        with torch.no_grad():
            for bi, batch in enumerate(dataloader):
                for key, val in batch.items(): batch[key] = val.to(self.device)                
                # 获取一个batch的样本表示 
                sample_resps = self.encode(batch)
                # 将样本表示向量按索引存储起来
                for idx, lab, resp in zip(batch['index'], batch['label'], sample_resps):
                    self.train_labels[idx.item()] = lab.item()
                    self.train_resps[idx.item()] = resp.detach().cpu()
                #process_tqdm.update()
        #process_tqdm.close()
        
    def forward(self, epoch, dataset, desc='train'):
        """
        1. 样本聚类统计
        2. 根据距离类中心距离, 获得样本难度排序
        3. 按概率选择参与训练的样本, 随着epoch增大, 困难样本选择概率变大
        """

        ## 0. 获取样本表示
        self.gen_representation(dataset) # 获取样本表示, 各类包含样本信息
        
        ## 1. 各类包含的表示取均值作为聚类中心
        self.centers = {}
        for label in range(self.n_class):
            index_mask = self.train_labels == label
            self.centers[label] = torch.stack(list(self.train_resps[index_mask])).mean(dim=0) # 计算类中心
            self.pools[label].append(self.centers[label]) # 类中心存入类池中

        ## 2. 计算样本难度, 执行课程学习
        if self.args.params_model.use_cl: # 执行课程学习
            ## 2. 根据个样本距离类中心距离，获得样本困难排序
            index_list = self.samples_rank()
            # 2.1. 按概率选择参加训练的样本 (刚开始困难样本小概率，逐渐增大困难样本的选择概率)
            st, ed = 1-epoch/self.args.params_train.epochs, epoch/self.args.params_train.epochs
            prob_list = torch.tensor([st+(ed-st)/(len(index_list)-1)*i for i in range(len(index_list))])
            indexes_mask = torch.bernoulli(prob_list).bool()
            trainset = [dataset.datas['vector']['train'][idx] for idx, mask in zip(index_list, indexes_mask) if mask]
            # 2.2 更新参数
            dataset.datas['vector']['train'] = trainset

        return dataset

    def samples_rank(self):
        def dist(x, y): return (1-F.cosine_similarity(x, y, dim=-1))/2

        scores = copy.deepcopy(self.train_labels)
        center_resps = torch.stack([resp for resp in self.centers.values()])
        for label in range(self.n_class):
            center_resp = self.centers[label]
            index_mask = self.train_labels==label
            resps = torch.stack(list(self.train_resps[index_mask]))
            score_label = dist(resps, center_resp) # 对自身所属label的相似得分
            score_labels = dist(resps.unsqueeze(dim=1), center_resps) # 对所有labels的相似得分
            scores[index_mask] = score_label / score_labels.sum(dim=-1)

        ranks = np.argsort(scores) # 升序, 越靠近中心越容易
        return ranks

    def similarity(self, resps_0, resps_1=None):
        """
        计算两组向量的相似性
        """
        if resps_1 is None: resps_1 = torch.clone(resps_0)

        shape_0, shape_1 = resps_0.shape[0], resps_1.shape[0] # 确定变化维度
        resps_0 = resps_0.unsqueeze(dim=1).expand(shape_0, shape_1, -1)
        resps_1 = resps_1.unsqueeze(dim=0).expand(shape_0, shape_1, -1)

        scores = (1 + F.cosine_similarity(resps_0, resps_1, dim=-1)) / 2
        return scores

    def ret_samples(self):
        # 获取各类暂时中心表示作为额外样本
        ret_resps, ret_resps_lab = [], []
        for lab, resps in self.pools.items():
            if len(resps) > self.pool_size_k: # 随机选择k个取均值
                resps = torch.stack(self.pools[lab], 0)
                select = torch.randperm(resps.size(0))[:self.pool_size_k]
                ret_resps.append(resps[select].mean(0))
            else: # 选原有的中心
                ret_resps.append(self.centers[lab])
            ret_resps_lab.append(lab)
        return  torch.stack(ret_resps), torch.tensor(ret_resps_lab)

    def scl_loss(self, features, labels, preds):
        ## 0. 将样本存储到所属类中
        for pred, lab, resp in zip(preds, labels, features):
            if pred == lab:
                self.pools[lab.item()].append(resp.detach().cpu()) # 预测正确就存储起来
            if len(self.pools[lab.item()]) > self.pool_size: # 保留最新的若干个
                self.pools[lab.item()] = self.pools[lab.item()][-self.pool_size:]

        ## 1. 获取各类暂时中心表示作为额外样本
        ret_features, ret_labels = self.ret_samples() # 额外样本 
        features_add = torch.cat((features, ret_features.type_as(features)), dim=0)
        labels_add = torch.cat((labels, ret_labels.type_as(labels)), dim=0)
        batch_size, batch_size_add = len(labels), len(labels_add)
        ## 2. 对比学习损失
        # 2.1 样本相似性得分 (排除了自身对)
        features_add_sim = self.similarity(features_add) / self.temp # 公式(4)
        scores = features_add_sim*(torch.ones_like(features_add_sim)-torch.eye(batch_size_add).type_as(labels))
        scores = torch.exp(scores-torch.max(scores).item())
        # 2.2 正负样本标记 (排除了自身对)
        labels_add_expand_0 = labels_add.unsqueeze(dim=0).expand(batch_size_add, batch_size_add)
        labels_add_expand_1 = labels_add.unsqueeze(dim=1).expand(batch_size_add, batch_size_add)
        label_mask = (labels_add_expand_0==labels_add_expand_1).long()
        label_pos_mask = label_mask - torch.eye(batch_size_add).type_as(labels)
        label_neg_mask = torch.ones_like(label_mask) - label_mask
        # 2.3 正负样本得分 (排除了自身对、排除了检索样本自身对)
        scores_pos = (scores * label_pos_mask).sum(dim=-1)[0:batch_size] # 公式(11)
        scores_neg = (scores * label_neg_mask).sum(dim=-1)[0:batch_size] # 公式(12)
        loss_scl = -torch.log((scores_pos/(scores_pos+scores_neg)/(label_pos_mask.sum(dim=-1)[0:batch_size]+self.eps))) # 公式(13)
        # # 2.3 正负样本得分 (排除了自身对、不排除检索样本自身对)
        # scores_pos = (scores * label_pos_mask).sum(dim=-1) # 公式(11)
        # scores_neg = (scores * label_neg_mask).sum(dim=-1) # 公式(12)
        # loss_scl = -torch.log((scores_pos/(scores_pos+scores_neg)/(label_pos_mask.sum(dim=-1)+self.eps))) # 公式(13)
        
        assert(loss_scl < 0).sum() == 0
        return loss_scl.mean()