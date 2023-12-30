import torch, os, fitlog, pickle, json
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn import CrossEntropyLoss, TransformerEncoderLayer
from transformers import BartPretrainedModel, BartModel, AutoTokenizer, AutoConfig
from config import random_parameters
from utils.dataloader_erc import *

"""
| dataset     | meld  | iec   | emn   | ddg   |

| baseline    | 83.42 | 79.36 | 00.00 |
| performance | 00.00 | 00.00 | 00.00 |

"""
baselines = {
    'base': {'meld': 0., 'iec': 0., 'emn': 0., 'ddg': 0}, 
    'large': {'meld': 0., 'iec': 0., 'emn': 0., 'ddg': 0},
}


class ERCDatasetCoG(ERCDataset_Multi):
    def __init__(self, args, tokenizer=None):
        self.args = args
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_seq_len
        self.container_init() # 初始化容器信息
        for desc, path in args.data_file.items():
            self.get_dataset(path, desc) # 解析数据集
        self.n_class = len(self.labels['ltoi']) 

    def refine_tokenizer(self, tokenizer=None):
        if tokenizer is None: tokenizer = self.tokenizer
        for token in self.tokens_add:
            tokenizer.add_tokens(token)

    def refine_utterance(self, utterance):
        if self.args.anonymity:
            speaker = self.speakers['ntoa'][utterance['speaker'].strip()] # speaker 匿名
        else:
            speaker = utterance['speaker'].strip() # speaker 不匿名
        speaker_text = speaker + ': ' + utterance['text'].strip()
        
        return {
            'text': utterance['text'].strip(), # speaker_text,
            'speaker': utterance['speaker'].strip(), # speaker,
            'label': utterance['label'].strip()
        }

    def get_sample(self, utterances):
        texts, speakers, labels = [], [], []
        for utt in utterances:
            text, speaker = utt['text'].strip(), utt['speaker'].strip() 
            label = 'none' if 'label' not in utt else utt['label'].strip()
            texts.append(text)
            speakers.append(speaker)
            labels.append(label)

        return {
            'texts': texts,
            'speakers': speakers,
            'labels': labels,
        }

    def get_dataset(self, path, desc):
        with open(path, 'r', encoding='utf-8') as fr:
            raw_data, samples = json.load(fr), []
            for di, dialog in enumerate(raw_data):
                utterances, speakers, labels = [], [], [] 
                for ui, utterance in enumerate(dialog):
                    self.speaker_label([utterance['speaker']], [utterance['label']])
                    utterance = self.refine_utterance(utterance)
                    utterances.append(utterance['text'].strip())
                    speakers.append(utterance['speaker'].strip())
                    labels.append(utterance['label'].strip())

                samples.append({
                    'index': di,
                    'texts': utterances,
                    'labels': labels,
                    'speakers': speakers
                })
        self.datas['text'][desc] = samples

    def get_vector(self, args=None, tokenizer=None, only=None):
        if args is None: args = self.args
        if args.anonymity: self.refine_tokenizer()
        if tokenizer is None: tokenizer = self.tokenizer
        for desc, data in self.datas['text'].items():
            if only is not None and desc!=only: continue
            data_embed = []
            for item in data:
                embedding = tokenizer(item['texts'], max_length=self.max_seq_len, padding='max_length', return_tensors='pt')
                input_ids, attention_mask = self.vector_truncate(embedding, method='first')
                speakers = [self.speakers['ntoi'][speaker] for speaker in item['speakers']]
                labels = [self.labels['ltoi'][label] for label in item['labels']]
                item_embed = {
                    'index': item['index'],
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'speakers': torch.tensor(speakers),
                    'labels': torch.tensor(labels),
                }
                data_embed.append(item_embed)

            self.datas['vector'][desc] = data_embed

    def collate_fn(self, dialogs):
        # 确定utterance最大token长度
        if 'attention_mask' in self.batch_cols.keys():
            max_token_len = max([torch.sum(dialog['attention_mask'], dim=-1).max() for dialog in dialogs]).item()
            max_utt_num = max([len(dialog['speakers']) for dialog in dialogs])

        ## 获取 batch
        inputs = {}
        for col, pad in self.batch_cols.items():
            if 'index' in col: temp = torch.tensor([dialog[col] for dialog in dialogs])
            if 'ids' in col or 'mask' in col: 
                temp = pad_sequence([dialog[col] for dialog in dialogs], batch_first=True, padding_value=pad)[:,:,0:max_token_len]
            if 'speakers' in col or 'labels' in col:
                temp = pad_sequence([dialog[col] for dialog in dialogs], batch_first=True, padding_value=pad)
            inputs[col] = temp

        return inputs


def config_for_model(args, scale='base'):
    scale = args.params_model.scale
    args.params_model.about = 'bart'
    args.params_model.plm = f'facebook/bart-{scale}'
    
    args.params_model.data = args.params_file.data + f'dataset.{args.params_model.name[0]}.' + scale
    args.params_model.baseline = baselines[scale][args.task[-1]]

    args.params_model.tokenizer = None
    args.params_model.optim_sched = ['AdamW', 'linear']

    args.params_model.use_generation = 0
    args.params_model.use_trans_layer = 0
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
        data_fn=ERCDataset_Multi,
    ) # for dataloader_absa
    dataset.batch_cols = {
        #'index': -1,
        'input_ids': dataset.tokenizer.pad_token_id, 
        'attention_mask': 0, 
        'speakers': -1, 
        'labels': -1, 
    }
    dataset.shuffle = {'train': True, 'valid': False, 'test': False}
    args.dataset = dataset

    # model = BartForERC(args) # 
    config = AutoConfig.from_pretrained(
        args.params_model.plm, 
        num_labels=args.dataset.n_class, 
        finetuning_task=args.task[-1]
    )
    model = BartForERC.from_pretrained(
        args.params_model.plm, 
        config=config, 
        args=args
    )

    return model


class TransformerUnit(nn.Module):

    def __init__(self, d_model: int, n_heads: int = 8):
        super(TransformerUnit, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        # self.out_features = out_features
        # activation by default the GELU
        self.transformerlayer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            activation='gelu'
        )

    def forward(self, features):
        features = self.transformerlayer(features)  # (bsz, dim)
        # dense_features = self.linear(features)  # (bsz, cls_num)
        return features

# class BartForERC(nn.Module):
#     def __init__(self, args, dataloader):
#         super().__init__()
class BartForERC(BartPretrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.params = args.params_model
        self.config = config

        self.plm_model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.plm_model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.plm_model.shared.num_embeddings, bias=False)
        self.init_weights()
        self.loss_ce = CrossEntropyLoss(ignore_index=-1)
        self.transformer_unit = TransformerUnit(d_model=config.hidden_size, n_heads=8)
        self.ffn = nn.Sequential(nn.Linear(config.hidden_size, 400),
                                 nn.Dropout(0.3),
                                 nn.GELU(),
                                 nn.Linear(400, config.num_labels))
        self.temperature = 5
        self.alpha = 0.2
        self.beta = 0.1
        self.num_labels = config.num_labels

    def shift_tokens_right(self, input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.plm_model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

    def forward(self, inputs, stage='train'):
        input_ids, attention_mask, speakers, labels = inputs['input_ids'], inputs['attention_mask'], inputs['speakers'], inputs['labels']
        # dialogue 参数统计
        utt_mask = torch.sum(inputs['attention_mask'], dim=-1) > 0 # utt-level mask
        # utt_mask = labels >= 0 # utt-level mask (attention_mask: token-level mask)
        batch_size, max_utt_len, max_token_len = input_ids.shape
        utt_lens = torch.sum(utt_mask, dim=-1) # 每个对话的utt长度
        # 经过bart得到每个utt的cls+hidden_state表示
        utt_bart = self.plm_model(input_ids=input_ids[utt_mask, :], attention_mask=attention_mask[utt_mask, :]) 
        utt_hidden_states = utt_bart.last_hidden_state # [utt_lens(sum), max_token_len, dim]
        # mask不存在的token，获得每个utt的表示
        mask_for_fill = attention_mask[utt_mask, :].unsqueeze(-1).expand(-1, -1, utt_hidden_states.shape[-1]).bool()
        utt_hidden_states_mask = utt_hidden_states.masked_fill(~mask_for_fill, -1e8)  # 实际不存在的token赋值-1e8
        utt_cls, _ = torch.max(utt_hidden_states_mask, dim=1)  # max pooling, 获得每个utt的表示
        
        # 恢复batch_size结构(utt-level)
        we_dim = self.config.hidden_size
        for bi in range(batch_size):
            temp = torch.zeros([max_utt_len-utt_lens[bi], we_dim], device=utt_cls.device) # 一个对话需要补充的东西，Pad不是1么？？
            position = bi*max_utt_len + utt_lens[bi]
            utt_cls = torch.cat([utt_cls[:position], temp, utt_cls[position:]], dim=0)
        utt_cls = utt_cls.view(batch_size, max_utt_len, we_dim)
        utt_cls_copy = utt_cls.clone().detach() # 复制一份，不参与梯度计算
        # 经过一个Transformer聚合上下文信息
        if self.params.use_trans_layer:
            utt_cls = self.transformer_unit(utt_cls)
            utt_cls_copy = self.transformer_unit(utt_cls_copy)

        # 经过线性层, 获得预测
        logits = self.ffn(utt_cls)
        logits_copy = self.ffn(utt_cls_copy)

        loss, loss_ce, loss_cl, loss_gen = 0, 0, 0, 0
        loss = self.loss_ce(logits[utt_mask, :], labels[utt_mask])
        if stage == 'train_':
            if self.params.use_generation:
                utt_mask_next = torch.sum(inputs['attention_mask_next'], dim=-1)>0
                decoder_input_ids = self.shift_tokens_right(inputs['input_ids_next'][utt_mask_next, :], self.config.pad_token_id, self.config.decoder_start_token_id)
                utt_gen_bart = self.plm_model(input_ids=input_ids[utt_mask, :], attention_mask=attention_mask[utt_mask, :], decoder_input_ids=decoder_input_ids)
                gen_hidden_state = utt_gen_bart.last_hidden_state
                logits_gen = self.lm_head(gen_hidden_state) + self.final_logits_bias
                loss_gen = self.loss_ce(logits_gen.view(-1, self.config.vocab_size), inputs['input_ids_next'][utt_mask_next, :].view(-1))

            loss_ce = self.loss_ce(logits[utt_mask, :], labels[utt_mask])
            loss_cl = SupConLoss(temperature=self.temperature,
                                 features=torch.stack([logits[utt_mask, :], logits_copy[utt_mask, :]], dim=1),
                                 labels=labels[utt_mask])
            loss = (1-self.alpha-self.beta)*loss_ce + self.alpha*loss_cl + self.beta*loss_gen
        
        
        return {
            'loss':     loss,
            'loss_ce':  loss_ce,
            'loss_cl':  loss_cl,
            'loss_gen': loss_gen,
            'preds': torch.argmax(logits[utt_mask, :], dim=-1).cpu(), 
            'labels': labels[utt_mask]
        }


def SupConLoss(temperature=0.07, contrast_mode='all', features=None, labels=None, mask=None):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""
    device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                         'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)  # 1 indicates two items belong to same class
    else:
        mask = mask.float().to(device)

    contrast_count = features.shape[1]  # num of views
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # (bsz * views, dim)
    if contrast_mode == 'one':
        anchor_feature = features[:, 0]
        anchor_count = 1
    elif contrast_mode == 'all':
        anchor_feature = contrast_feature  # (bsz * views, dim)
        anchor_count = contrast_count  # num of views
    else:
        raise ValueError('Unknown mode: {}'.format(contrast_mode))

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T), temperature)  # (bsz, bsz)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # (bsz, 1)
    logits = anchor_dot_contrast - logits_max.detach()  # (bsz, bsz) set max_value in logits to zero
    # logits = anchor_dot_contrast

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)  # (anchor_cnt * bsz, contrast_cnt * bsz)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
                                0)  # (anchor_cnt * bsz, contrast_cnt * bsz)
    mask = mask * logits_mask  # 1 indicates two items belong to same class and mask-out itself

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask  # (anchor_cnt * bsz, contrast_cnt * bsz)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

    # compute mean of log-likelihood over positive
    if 0 in mask.sum(1):
        raise ValueError('Make sure there are at least two instances with the same class')
    # temp = mask.sum(1)
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

    # loss
    # loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
    loss = -mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()
    return loss