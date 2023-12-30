import torch, os, copy, fitlog, math, json
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset



class MMDataset(Dataset):
    def __init__(self, path, lower=False):
        self.path = path
        self.lower = lower
        self.name = ['med', path.split('/')[-2]]
        # self.flow_num = flow_nums[self.name[-1]] 
        self.max_seq_len = 128 # max_seq_lens[self.name[-1]] 
        self.container_init() # 初始化容器信息
        for desc in ['train', 'valid', 'test']:
            self.datas['source'][desc.split('_')[0]] = self.get_dataset(path, desc)

        

    def container_init(self, only='all'):
        # 初始化数据集要保存的内容 
        path_tokenizer_ = self.path + 'tokenizer_'
        if os.path.exists(path_tokenizer_):
            self.tokenizer_ = torch.load(path_tokenizer_)
            self.path_tokenizer_ = None
        else:
            self.tokenizer_ = {
                'labels': { 'ltoi': {}, 'itol': {}, 'count': {}, 'class': [] }, # 分类标签字典
            }
            self.path_tokenizer_ = path_tokenizer_

        self.datas = {
            'source': {},        # 解析后纯文本
            'vector': {},      # 文本分词后数字化表示
            'dataloader': {},  # dataloader用于构建batch
        }

    def get_dataset(self, path, desc='train'):
        dirs, samples = os.listdir(path), []
        for d in dirs:
            files = os.listdir(os.path.join(path, d))
            for file in files:
                sample = {
                    'index': len(samples),
                    'image': os.path.join(path, d, file),
                    'text': d,
                }
                samples.append(sample)

        return samples

    def get_vector(self, tokenizer, truncate=None, only=None):
        self.tokenizer = tokenizer
        self.add_tokens() # 将重构的speaker、label添加到字典
        sep_token, mask_token = tokenizer.sep_token, tokenizer.mask_token
        for stage, dialogs in self.datas['text'].items():
            if only is not None and stage!=only: continue
            samples, dialog_tokens_num, dialog_speakers_num = [], [], []
            for dialog in tqdm(dialogs):
                new_dialog = {'index': dialog['index'], 'texts': [], 'speakers': [], 'polarities': [], 'labels': []}
                embedding = {'input_ids': [], 'attention_mask': [], 'adj': []}
                for ui in range(len(dialog['texts'])):
                    # if dialog['labels'][ui] is None: continue
                    new_dialog['speakers'].append(dialog['speakers'][ui])
                    new_dialog['polarities'].append(dialog['labels'][ui]) 
                    lab_emo = self.tokenizer_['labels']['ltoi'][dialog['labels'][ui]] if dialog['labels'][ui] is not None else -1
                    new_dialog['labels'].append(lab_emo) 
                    
                    # utt_prompt = dialog['texts'][ui]
                    # for k in list(range(ui))[::-1]:
                    #     if dialog['speakers'][k] != dialog['speakers'][ui]: continue 
                    #     utt_prompt = dialog['texts'][k]+f' {sep_token} ' + utt_prompt
                    # utt_prompt = f"for '{utt_prompt}' {dialog['speakers'][ui]} feels {mask_token} {sep_token}"

                    utt_prompt = dialog['speakers'][ui]+': '+dialog['texts'][ui]+f' {sep_token} '+dialog['speakers'][ui]+f' displays {mask_token}'
                    for k in list(range(ui))[::-1]:
                        utt_prompt = dialog['speakers'][k]+': '+dialog['texts'][k]+f' {sep_token} ' + utt_prompt

                    #utt_prompt = dialog['speakers'][ui]+': '+dialog['texts'][ui]+f' {sep_token} '+dialog['speakers'][ui]+f' displays {mask_token}'
                    spk_tmp, adj_tmp = [], [0]*len(dialog['texts']) 
                    for k in list(range(ui))[::-1]:
                        if dialog['speakers'][k] not in spk_tmp and dialog['labels'] is not None:
                            spk_tmp.append(dialog['speakers'][k])
                            adj_tmp[k] = 1
                            #utt_prompt = dialog['speakers'][k]+': '+dialog['texts'][k]+f' {sep_token} ' + utt_prompt
                    adj_tmp[ui] = 1
                    embedding['adj'].append(adj_tmp)

                    input_ids = torch.tensor([tokenizer.cls_token_id]+self.tokenizer(utt_prompt, add_special_tokens=False)['input_ids'][-self.max_seq_len:])
                    embedding['input_ids'].append(input_ids)
                    embedding['attention_mask'].append(torch.ones_like(input_ids))

                    new_dialog['texts'].append(utt_prompt)

                self.info['utt_num'][stage].append(len(new_dialog['labels']))
                self.info['token_num'][stage].extend([len(utt.split()) for utt in new_dialog['texts']])
                # embedding = self.tokenizer(new_dialog['texts'], max_length=self.max_seq_len, padding='max_length', truncation=True, return_tensors='pt')
                #embedding = self.tokenizer(new_dialog['texts'], padding=True, return_tensors='pt')
                embedding['input_ids'] = pad_sequence(embedding['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id)
                embedding['attention_mask'] = pad_sequence(embedding['attention_mask'], batch_first=True, padding_value=tokenizer.pad_token_id)
                embedding['adj'] = torch.tensor(embedding['adj'])
                new_dialog['embedding'] = embedding
                dialog['new_dialog'] = new_dialog
                dialog = visible_matrix(dialog)

                dialog_tokens_num.append(embedding['attention_mask'].sum())
                dialog_speakers_num.append(len(set(new_dialog['speakers'])))

            self.info['dialog_tokens_num'][stage] = dialog_tokens_num
            self.info['dialog_speakers_num'][stage] = dialog_speakers_num
        

    def collate_fn(self, samples):
        dialogs, inputs = [sample['new_dialog'] for sample in samples], {}
        bz = len(dialogs)
        return {
            'input_ids': [dialogs[i]['embedding']['input_ids'] for i in range(bz)],
            'attention_mask': [dialogs[i]['embedding']['attention_mask'] for i in range(bz)],
            'adj': [dialogs[i]['embedding']['adj'] for i in range(bz)],
            'labels': [torch.tensor(dialogs[i]['labels']) for i in range(bz)],
            'visible': [torch.tensor(dialogs[i]['visible']) for i in range(bz)],
            'position': [torch.tensor(dialogs[i]['position']) for i in range(bz)],
        }

        max_utterance_len = max([len(dialog['labels']) for dialog in dialogs])
        for col, pad in self.batch_cols.items():
            if col == 'index':
                temp = torch.tensor([dialog[col] for dialog in dialogs])
            elif col == 'labels':
                temp = pad_sequence([dialog[col] for dialog in dialogs], batch_first=True, padding_value=pad)
            else:
                temp = []
                for dialog in dialogs:
                    col_tmp = dialog['embedding'][col]
                    col_tmp_add = torch.ones(max_utterance_len-col_tmp.shape[0], col_tmp.shape[1])*pad
                    temp.append(torch.cat([col_tmp, col_tmp_add]))
                temp = pad_sequence(temp, batch_first=True, padding_value=pad)

            inputs[col] = temp

        return inputs
    

def config_for_model(args):
    scale = args.model['scale']
    #args.model['plm'] = f'princeton-nlp/sup-simcse-roberta-{scale}'
    #args.model['plm'] = f'microsoft/deberta-{scale}'
    args.model['plm'] = f'roberta-{scale}'

    args.model['data'] = args.file['data'] + f"dataset.{args.model['name']}." + scale

    args.model['tokenizer'] = None
    args.model['optim_sched'] = ['AdamW_', 'cosine']
    #args.model['optim_sched'] = ['AdamW_', 'linear']

    return args
             
def import_model(args):
    ## 1. 更新参数
    args = config_for_model(args) # 添加模型参数, 获取任务数据集
    fitlog.set_rng_seed(args.train['seed'])
    
    ## 2. 导入数据
    data_path = args.model['data']
    if os.path.exists(data_path):
        dataset = torch.load(data_path)
    else:
        dataset = MMDataset(path=args.file['data'], lower=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model['plm'])
        dataset.get_vector(tokenizer, truncate='first')
        torch.save(dataset, args.model['data'])

    ## 3. 统计情绪转移
    dataset.datas['vector'] = dataset.datas['text']
    dataset.batch_cols = {
        'index': -1,
        'input_ids': dataset.tokenizer.pad_token_id,
        # 'attention_mask': 0,
        'labels': -1,
        'visible': 0,
        'position': 0,
    }
    dataset.shuffle = {'train': True, 'valid': False, 'test': False}

    model = EFCL(
        args=args,
        dataset=dataset,
        plm=args.model['plm'],
    )

    return model, dataset

class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0):
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:  # dot_product / scaled_dot_product
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, q, k, v=None, mask=None, wei=None):
        if len(q.shape) == 1:  # q_len missing
            q = torch.unsqueeze(q, dim=0)
        assert len(k.shape) == 2
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head*?, k_len, hidden_dim)
        # qx: (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        kx, qx = self.w_k(k), self.w_q(q)
        if self.score_function == 'dot_product':
            kt = kx.permute(1, 0)
            score = torch.matmul(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            # kq = torch.unsqueeze(kx, dim=1) + torch.unsqueeze(qx, dim=2)
            score = F.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')

        if mask is not None: 
            score = torch.add(score, (mask.reshape(score.shape).int()-1)*1e16)
        score = F.softmax(score, dim=-1)
        if wei is not None:
            score = torch.mul(score, wei.type_as(score))
            score = torch.add(score, (mask.reshape(score.shape).int()-1)*1e16)
            score = F.softmax(score, dim=-1)
        output = torch.matmul(score, kx)  # (n_head*?, q_len, hidden_dim)
        #output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (?, q_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        return output.squeeze(dim=0), score.squeeze(dim=0)
    

class PoolerAll(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states # [:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class EFCL(nn.Module):
    def __init__(self, args, dataset, plm=None):
        super().__init__()
        self.args = args
        self.n_class = dataset.n_class
        self.mask_token_id = dataset.tokenizer.mask_token_id

        self.plm_model = AutoModel.from_pretrained(plm, local_files_only=False) 
        self.vocab_size = len(dataset.tokenizer)
        self.plm_model.resize_token_embeddings(self.vocab_size)

        self.plm_model.pooler_all = PoolerAll(self.plm_model.config)
        self.hidden_dim = self.plm_model.config.hidden_size   

        self.gat = GAT(
            nfeat=self.hidden_dim, 
            nhid=self.hidden_dim//8, 
            nclass=self.hidden_dim, 
            dropout=0.3, 
            nheads=8, 
            alpha=0.2)
        self.position_embeddings = nn.Embedding(100, self.hidden_dim)
        self.attn = Attention(self.hidden_dim)
        self.dropout = nn.Dropout(args.model['drop_rate'])
        self.classifier = nn.Linear(self.hidden_dim, self.n_class)
        self.linear_0 = nn.Linear(self.hidden_dim*2, self.hidden_dim)

        self.loss_ce = CrossEntropyLoss(ignore_index=-1)
        self.loss_sce = LabelSmoothSoftmaxCEV1(lb_smooth=None, reduction='weight', ignore_index=-1)

        self.hmm_P = (torch.ones(self.n_class)*1/self.n_class).unsqueeze(dim=1)
        self.hmm_A = dataset.info['hmm_A'] # [nn.Sequential(nn.Linear(self.hidden_dim, 7), nn.Softmax()).to('cuda') for _ in range(7)]
        self.hmm_B = nn.Sequential(
            self.dropout,
            self.classifier, #nn.Linear(self.hidden_dim, n_class), 
            nn.Softmax()
        ) # 需要归一化
        # self.linear_1 = PoolerAll(self.plm_model.config) # nn.Linear(self.hidden_dim, self.hidden_dim)
        # self.linear_2 = PoolerAll(self.plm_model.config) # nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_2 = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.print_trainable_parameters(self) # 参与训练的参数
   
    def print_trainable_parameters(self, model):
            """
            Prints the number of trainable parameters in the model.
            """
            trainable_params = 0
            all_param = 0
            for _, param in model.named_parameters():
                num_params = param.numel()
                # if using DS Zero 3 and the weights are initialized empty
                if num_params == 0 and hasattr(param, "ds_numel"):
                    num_params = param.ds_numel

                all_param += num_params
                if param.requires_grad:
                    trainable_params += num_params
            print(
                f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
            )

    def epoch_deal(self, epoch, dataset, device='cpu'): # 针对的是训练集
        # 训练集改变了, 重新统计aspect信息
        self.eval()
        if None in self.bank:
            train_dataloader = dataset.get_dataloader(24, only='train')['train']
            for batch in train_dataloader:
                for key, val in batch.items(): batch[key] = val.to(device)
                with torch.no_grad():
                    clss, _, _ = self.encode(batch)     
                    self.get_features(batch['index'], features=clss, method='store')

    def encode(self, inputs):
        plm_out = self.plm_model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            output_hidden_states=True,
            return_dict=True
        )
        hidden_states = self.plm_model.pooler_all(plm_out.last_hidden_state)

        # 取出 mask 表示
        mask_bool = inputs['input_ids'] == self.mask_token_id
        hidden_states_mask, loss_mlm = hidden_states[mask_bool], 0

        return hidden_states[:, 0], hidden_states_mask

    def get_wei(self, pos, t_num):
        mask_ = torch.triu(torch.ones(t_num,t_num), diagonal=0) * (pos.max(dim=1)[0]+1).unsqueeze(dim=1)
        adj = pos + mask_
        adj = (pos.max(dim=1)[0]+1).unsqueeze(dim=1) - adj
        adj = adj / adj.sum(dim=1).unsqueeze(dim=1)

        return adj

    def diff_loss(self, input1, input2):

        # input1 (B,N,D)    input2 (B,N,D)
        batch_size = input1.size(0)
        N = input1.size(1)
        input1 = input1.view(batch_size, -1)  # (B,N*D)
        input2 = input2.view(batch_size, -1)  # (B, N*D)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True) # (1,N*D)
        input2_mean = torch.mean(input2, dim=0, keepdims=True) # (1,N*D)
        input1 = input1 - input1_mean     # (B,N*D)
        input2 = input2 - input2_mean     # (B,N*D)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach() # (B,1)
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6) # (B,N*D)
        
        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach() # (B,1)
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6) # (B,N*D)

        diff_loss = 1.0/(torch.mean(torch.norm(input1_l2-input2_l2,p=2,dim=1)))
  
        return diff_loss

    def hmm(self, obv):
        log_hmm = []
        for fi, fea in enumerate(obv):
            if fi == 0:
                bb = self.hmm_B(fea).reshape_as(self.hmm_P)
                p = self.hmm_P.type_as(fea)*bb
                p = p*1/sum(p)
            else:
                p = torch.mm(self.hmm_A.type_as(fea).T, log_hmm[-1].T) * self.hmm_B(fea).reshape_as(self.hmm_P)
                p *= 1/sum(p)
            log_hmm.append(p.transpose(1,0))

        return log_hmm

    def forward(self, inputs, stage='train'):
        bz = len(inputs['input_ids'])
        features, labels, log_cls, log_hmm, log_hmm_add, labels_hmm_add, diff_loss = [], [], [], [], [], [], []
        for i in range(bz):
            input_tmp = { }
            for k, v in inputs.items():
                input_tmp[k] = v[i].to(self.args.train['device'])
            fea_cls, fea_emo = self.encode(input_tmp)
            #diff_loss.append(self.diff_loss(fea_cls, fea_emo))

            fea_cls_hmm = fea_emo
            ## emo 作 HMM
            # for fi, fea in enumerate(fea_cls_hmm):
            #     if fi == 0:
            #         p = self.hmm_P.type_as(fea)*self.hmm_B(fea)
            #         p = p*1/sum(p)
            #     else:
            #         temp = (log_hmm[-1]*self.hmm_A.type_as(fea)).sum(dim=1)
            #         p = torch.tensor(temp).type_as(fea)*self.hmm_B(fea)
            #         p *= 1/sum(p)
            #     log_hmm.append(p)
            
            for fi, adj in enumerate(input_tmp['adj']):
                p = self.hmm(fea_emo[adj.bool()])
                log_hmm_add.extend(p)
                labels_hmm_add.extend(copy.deepcopy(input_tmp['labels'][adj.bool()]))
                log_hmm.append(p[-1])

            # mask = torch.tril(torch.ones(fea_cls.shape[0],fea_cls.shape[0]),diagonal=0)-torch.eye(fea_cls.shape[0],fea_cls.shape[0])
            for _ in range(1):
                fea_contextual = self.gat(fea_cls, input_tmp['adj']-torch.eye(len(fea_cls)).type_as(fea_cls))
                fea_contextual = torch.cat([self.dropout(fea_cls)[0:1], fea_contextual[1:]])
                fea_cls_contextual = self.linear_0(torch.cat([fea_cls, self.linear_1(fea_contextual)], dim=-1))
                fea_cls = fea_cls_contextual

            log_cls.append(F.softmax(self.classifier(self.dropout(fea_cls_contextual))))

            ## cls 作 Attention
            # fea_cls_attn = fea_cls
            # temp = fea_cls_attn.unsqueeze(dim=0).repeat(fea_cls_attn.shape[0],1,1)# +self.position_embeddings(inputs['position'][i].to('cuda'))
            # mask = torch.tril(torch.ones(fea_cls_attn.shape[0],fea_cls_attn.shape[0]),diagonal=0)-torch.eye(fea_cls_attn.shape[0],fea_cls_attn.shape[0])
            # wei = self.get_wei(inputs['position'][i], fea_cls.shape[0])
            # for k in range(len(mask)):
            #     if k == 0:
            #         fea_contextual = fea_cls[k] # 没有就给自己
            #     else:
            #         fea_cls_attn_tmp, _ = self.attn(temp[k][k], temp[k], mask=mask[k].type_as(temp).unsqueeze(dim=0), wei=wei[k])
            #         fea_cls_attn_con = (fea_cls_attn_tmp+temp[k][k])/2
            #     features.append(fea_cls_attn_con)
            #     log_cls.append(F.softmax(self.classifier(self.dropout(fea_cls_attn_con.squeeze(dim=0)))))
                
            labels.append(input_tmp['labels'])
        labels = torch.cat(labels)
        # logits = torch.log(torch.stack(log_cls)*0.5+torch.stack(log_hmm)*0.5)
        logits = torch.log(torch.cat(log_cls, dim=0) + torch.cat(log_hmm))
        lab_mask = labels >= 0
        labels, logits = labels[lab_mask], logits[lab_mask]
        assert -1 not in labels

        lb_one_hot = torch.empty_like(logits).fill_(0).scatter_(1, labels.unsqueeze(1), 1).detach()
        loss = -torch.sum(logits * lb_one_hot, dim=1).mean()

        # logits_add = torch.log(torch.cat(log_hmm_add))
        # labels_add = torch.stack(labels_hmm_add)
        # lab_mask = labels_add >= 0
        # labels_add, logits_add = labels_add[lab_mask], logits_add[lab_mask]
        # assert -1 not in labels_add
        # lb_one_hot_add = torch.empty_like(logits_add).fill_(0).scatter_(1, labels_add.unsqueeze(1), 1).detach()
        # loss_add = -torch.sum(logits_add * lb_one_hot_add, dim=1).mean()
        # loss = loss*0.8 + loss_add*0.2 #+ sum(diff_loss)/len(diff_loss)

        # wei = F.softmax(torch.tensor([(len(labels)-i)/len(labels) for i in range(len(labels))])).to('cuda').detach()
        # loss = -torch.sum(logits * lb_one_hot * wei.unsqueeze(dim=1), dim=1).sum()
        preds = torch.argmax(logits, dim=-1).cpu()

        # features = torch.stack(features)
        # labels = torch.cat(labels)
        # logits = self.classifier(self.dropout(features))
        # preds = torch.argmax(logits, dim=-1).cpu()
        # loss = self.loss_ce(logits, labels)

        return {
            'loss':   loss,
            'logits': logits,
            'preds':  preds,
            'labels': labels,
        }
    
    def get_features(self, index, features, method='fetch'):
        # 根据 index 存储/获得 实例表示向量
        for bi, idx in enumerate(index):
            if method == 'store': # 存储表示
                self.bank[idx] = features[bi].detach().cpu()
            if method == 'fetch': # 取出表示
                features.append(torch.stack([fea for fea in self.bank[idx.cpu().numpy()] if fea is not None]).mean(dim=0))
        
        assert len(features) == len(index)
        return features

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

    def fl_loss(self, features, labels):
        ## 情绪流中的mask表示靠近情绪标签词
        plm_out = self.plm_model(
            input_ids=self.labels_pro['ids'].type_as(labels),
            attention_mask=self.labels_pro['mask'].type_as(labels),
            output_hidden_states=True,
            return_dict=True
        )
        hidden_states = self.plm_model.pooler(plm_out.last_hidden_state).mean(dim=1)
        sim = self.contrast_single(features, hidden_states)
        return self.loss_ce(sim, labels)

        ## 情绪流中的mask表示靠近lab原型
        # lab_features = []
        # for lab in range(self.n_class):
        #     lab_features.append(torch.stack([fea for fea in self.bank[self.labels_mask[lab]]]).mean(dim=0))
        
        # sim = self.contrast_single(features, torch.stack(lab_features).type_as(features))
        # return self.loss_ce(sim, labels)


        ## scl 损失
        features_add = torch.cat([features, hidden_states], dim=0)
        labels_add = torch.cat([labels, torch.tensor(list(range(self.n_class))).type_as(labels)], dim=0)
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

    def cl_loss(self, features, inputs):
        index, labels = inputs['index'], inputs['label']
        # 存储样本表示
        self.get_features(inputs['index'], features=features, method='store') # 存储表示
        # 获取检索样本表示
        ret_features = self.get_features(inputs['ret_idx'], features=[], method='fetch')    
        ret_features = self.dropout(torch.stack(ret_features).type_as(features))
        
        # Contrast Learning 
        eye_mask = torch.eye(features.shape[0]).type_as(features)
        con_features = torch.cat([ret_features, features]) # 原来的batch也要in-batch负采样
        sample_mask = torch.cat([torch.zeros_like(eye_mask), eye_mask], dim=-1) # 定位原本feature位置

        ## 正样本：检索正样本(1个); 负样本：检索负样本(若干)+in-batch负样本(若干)
        labels_cl = torch.arange(features.size(0)).type_as(labels)
        # loss_cl = self.contrast_single(features_tar, features_scl, labels_scl)
        sim_tar_all = self.contrast_single(features, con_features)
        sim_tar_all = sim_tar_all - sample_mask*1e8
        loss_cl = self.loss_ce(sim_tar_all, labels_cl)
        
        return loss_cl/(features.shape[-1]**0.25)


        features_add = torch.cat((features, torch.stack(ret_features).type_as(features)), dim=0)
        labels_add = torch.cat((labels, labels), dim=0)
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