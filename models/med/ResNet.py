import torch, os, copy, fitlog, math, json
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from utils.processor_utils import set_rng_seed
from utils.metric_seg import WeightedDiceBCE
from models.med.LViT import iou_on_batch
import segmentation_models_pytorch as smp

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
    args.model['plm'] = args.file['plm'] + f'roberta-{scale}'

    args.model['data'] = args.file['data'] + f"dataset.{args.model['name']}." + scale

    args.model['tokenizer'] = None
    args.model['optim_sched'] = ['AdamW_', 'cosine']
    #args.model['optim_sched'] = ['AdamW_', 'linear']

    return args
             
def import_model(args):
    ## 1. 更新参数
    args = config_for_model(args) # 添加模型参数, 获取任务数据集
    set_rng_seed(args.train['seed'])
    
    ## 2. 导入数据
    args.train['data_rate'] = 0.01
    if args.train['data'] == 'Covid19':
        from datasets.med.Covid19.data_loader import get_specific_dataset
        dataset = get_specific_dataset(args)
    if args.train['data'] == 'Ptery':
        from datasets.med.Ptery.data_loader import get_specific_dataset
        dataset = get_specific_dataset(args)

    ## 3. 导入模型
    model = _ResNet(args, dataset)
    return model, dataset


class _ResNet(nn.Module):
    def __init__(self, args, dataset, plm=None):
        super().__init__()
        self.args = args
        self.dataset = dataset
        # self.plm_model = models.resnet50(pretrained=True)
        # self.plm_model = models.segmentation.fcn_resnet50(pretrained=True)
        # self.plm_model = smp.Unet('resnet34', encoder_weights='imagenet')
        self.plm_model = smp.Unet('resnet34', encoder_weights=None)
        self.conv2d = nn.Conv2d(21, 1, kernel_size=(1, 1), stride=(1, 1))
        self.activate = nn.Sigmoid()

        self.loss_bce = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)

    def forward(self, inputs, stage='train'):
        # output = self.plm_model(inputs['p_img'])['out']
        # logits = self.activate(self.conv2d(output))
        # loss = self.loss_bce(logits, inputs['p_lab'].float())

        output = self.plm_model(inputs['p_img'])
        logits = self.activate(output)
        loss = self.loss_bce(logits, inputs['p_lab'].float())

        return {
            'logits': logits,
            'loss': loss,
            # 'preds': None,
            'labels': inputs['p_lab'],
            # 'dice': self.loss_bce._show_dice(logits, lab.float()),
            # 'iou': iou_on_batch(lab, logits),
            'dice_fn': self.loss_bce._show_dice,
            'iou_fn': iou_on_batch
        }