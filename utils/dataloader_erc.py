import json, torch, os
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

"""
data_file = {
    'train': f'{args.dir_data}/{args.task_name}/train_data.json',
    'valid': f'{args.dir_data}/{args.task_name}/dev_data.json',
    'test' : f'{args.dir_data}/{args.task_name}/test_data.json',
}
"""

## tokenizer
def iterSupport(func, query):
    # 迭代处理 list 数据
    if isinstance(query, (list, tuple)):
        return [iterSupport(func, q) for q in query]
    try:
        return func(query)
    except TypeError:
        return func[query]

class Embedding(object):
    def __init__(self, inputs) -> None:
        self.input_ids = inputs['input_ids']
        self.attention_mask = inputs['attention_mask']
        self.attention_mask_asp = inputs['attention_mask_asp']

class Tokenizer(object):
    def __init__(self, embed_dim=None, lower=True, is_full=True):
        self.lower = lower
        self.embed_dim = embed_dim
        self.words = {}
        if is_full:
            self.pad_token  = '<pad>' # '[PAD]'
            self.unk_token  = '<unk>' # '[UNK]'
            self.mask_token = '<mask>' # '[MASK]'
            self.vocab = {self.pad_token: 0, self.unk_token: 1, self.mask_token: 2}
            self.ids_to_tokens = {0: self.pad_token, 1: self.unk_token, 2: self.mask_token}
        else:
            self.vocab = {}
            self.ids_to_tokens = {}
    
    def count(self, ele):
        if self.lower: ele = ele.lower()
        if ele not in self.words: self.words[ele] = 0
        self.words[ele] += 1

    def add_tokens(self, tokens):
        if not isinstance(tokens, list): tokens = [tokens]
        for token in tokens:
            if self.lower: token = token.lower()
            if token not in self.vocab: 
                self.vocab[token] = len(self.vocab)
                self.ids_to_tokens[len(self.ids_to_tokens)] = token
        self.vocab_size = len(self.vocab)

    def get_vocab(self, min_count=1):
        for ele, count in self.words.items():
            if count > min_count:
                self.vocab[ele] = len(self.vocab)
                self.ids_to_tokens[len(self.ids_to_tokens)] = ele
        self.vocab_size = len(self.vocab)
        
    def get_word_embedding(self):
        # glove_file = "E:/OneDrive - stu.xmu.edu.cn/MyStudy/My_Paper/Paper Reference (NLP)/Paper Coding/Coding_Tools/Glove/glove.840B/glove.840B.300d.txt"
        glove_file = "/home/jzq/glove/glove.840B.300d.txt"
        words_embed = {}
        with open(glove_file, 'r', encoding='utf-8', newline='\n', errors='ignore') as fr:
            for line in fr:
                tokens = line.rstrip().split()
                word = ''.join(tokens[0:len(tokens)-self.embed_dim])
                if word in self.vocab: words_embed[word] = [float(ele) for ele in tokens[-self.embed_dim:]]

        matrix = []
        for key, val in self.vocab.items():
            if key in words_embed:
                matrix.append([float(ele) for ele in words_embed[key]])
            elif key == self.pad_token:
                matrix.append([0]*self.embed_dim)
            else:
                matrix.append([random.uniform(0, 1) for i in range(self.embed_dim)])

        self.word_embedding = matrix

    def get_sent_embedding(self, sentence, method='mean'):
        # 输入一句话，输出一个语义向量
        idxs = self.tokens_to_ids(sentence.split(' '))
        embedding = torch.tensor([self.word_embedding[idx] for idx in idxs])
        if method == 'mean':
            return embedding.mean(dim=0)

    def tokens_to_ids(self, tokens):
        # 输入tokens，输出句子的 id 表示
        return torch.tensor([self.vocab[token] if token in self.vocab else self.vocab[self.unk_token] for token in tokens])

    def encode(self, texts, padding=True, return_tensors='pt', add_special_tokens=False):
        input_ids = []
        for text in texts:
            tokens = text.split(' ')
            input_ids.append(self.tokens_to_ids(tokens))
        pad_id = self.vocab[self.pad_token]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
        attention_mask = input_ids != pad_id

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

def get_tokenizer(path, tokens):
    ## 保存tokenizer
    if os.path.exists(path):
        tokenizer = torch.load(path)
    else:
        tokenizer = Tokenizer(embed_dim=300, lower=False)
        iterSupport(tokenizer.count, tokens) # 统计词频
        tokenizer.get_vocab(min_count=0) # 获得指向
        # tokenizer.get_word_embedding() # 获取embedding
        # torch.save(tokenizer, path)
        
    return tokenizer


class ERCDataset_Multi(Dataset):
    def __init__(self, data_path, lower=False):
        self.lower = lower
        self.data_path = data_path
        self.name = ['erc', data_path.split('/')[-2]]
        self.max_seq_len = 256
        self.container_init() # 初始化容器信息
        for desc in ['train', 'valid', 'test']:
            self.datas['text'][desc] = self.get_dataset(data_path, desc) # 解析数据集
        self.info['emotion_category'] = self.tokenizer_['labels']['ltoi']
        self.n_class = len(self.tokenizer_['labels']['ltoi']) 

    def container_init(self, only='all'):
        self.info = {
            'dialog_num': {'train': 0, 'valid': 0, 'test': 0}, # 数据对话数
            'utt_num': {'train': [], 'valid': [], 'test': []}, # dialog最大utterance数
            'token_num': {'train': [], 'valid': [], 'test': []}, # utterance最大token数
            'total_samples_num': {'train': 0, 'valid': 0, 'test': 0}, # 重构样本数量
            'dialog_tokens_num': {'train': 0, 'valid': 0, 'test': 0}, # 样本长度
            'dialog_speakers_num': {'train': 0, 'valid': 0, 'test': 0}, # speaker 人数
            'emotion_category': {}, # 情绪类别
        }
        # 初始化数据集要保存的内容 
        path_tokenizer_ = self.path + 'tokenizer_'
        if os.path.exists(path_tokenizer_):
            self.tokenizer_ = torch.load(path_tokenizer_)
            self.path_tokenizer_ = None
        else:
            self.tokenizer_ = {
                'labels': { 'ltoi': {}, 'itol': {}, 'count': {}, 'class': [] }, # 分类标签字典
                'speakers': { 'stoi': {}, 'itos': {}, 'count': {}}, # speaker字典
            }
            self.path_tokenizer_ = path_tokenizer_


        self.datas = {
            'text': {},        # 解析后纯文本
            'vector': {},   # 文本分词后数字化表示
            'dataloader': {},  # dataloader用于构建batch
        }
        self.tokens_add = [] # speaker 匿名token

    def speaker_label(self, speakers, labels):
        if self.path_tokenizer_ is None: return -1

        # 记录speaker信息
        for speaker in speakers:
            if speaker not in self.tokenizer_['speakers']['stoi']: # 尚未记录
                self.tokenizer_['speakers']['stoi'][speaker] = len(self.tokenizer_['speakers']['stoi'])
                self.tokenizer_['speakers']['itos'][len(self.tokenizer_['speakers']['itos'])] = speaker
                self.tokenizer_['speakers']['count'][speaker] = 1
            self.tokenizer_['speakers']['count'][speaker] += 1

        # 记录label信息
        for label in labels:
            if label is None: continue
            if label not in self.tokenizer_['labels']['ltoi']:
                self.tokenizer_['labels']['ltoi'][label] = len(self.tokenizer_['labels']['ltoi'])
                self.tokenizer_['labels']['itol'][len(self.tokenizer_['labels']['itol'])] = label
                self.tokenizer_['labels']['class'].append(label)
                self.tokenizer_['labels']['count'][label] = 1
            self.tokenizer_['labels']['count'][label] += 1

    def get_sample(self, utterances):
        texts, speakers, labels = [], [], []
        for utt in utterances:
            text, speaker = utt['text'].strip(), utt['speaker'].strip() 
            label = utt.get('label')
            texts.append(text)
            speakers.append(speaker)
            labels.append(label)

        return {
            'index': 0,
            'texts': texts,
            'speakers': speakers,
            'labels': labels,
        }

    def get_dataset(self, path, desc):
        raw_data_path = f'{path}/{desc}.raw.json'
        with open(raw_data_path, 'r', encoding='utf-8') as fp:
            raw_samples, samples = json.load(fp), []
        self.info['dialog_num'][desc] = len(raw_samples)

        for di, utts in enumerate(raw_samples):
            speakers = [utt['speaker'].strip() for utt in utts]
            labels = [utt.get('label') for utt in utts]
            assert len(speakers) == len(labels)
            self.speaker_label(speakers, labels) # 记录speakers/labels
            sample = self.get_sample(utts)
            sample['index'] = len(samples)
            samples.append(sample)

        #self.datas['text'][desc] = samples
        #if self.path_tokenizer_: torch.save(self.tokenizer_, self.path_tokenizer_)
        return samples 

    def vector_truncate(self, embedding, truncate='tail'):
        input_ids, attention_mask = embedding['input_ids'], embedding['attention_mask']
        cur_max_seq_len = max(torch.sum(attention_mask, dim=-1)) # 当前最大句子长度
        if cur_max_seq_len > self.max_seq_len:
            if truncate == 'tail': # 截断后面的
                temp = input_ids[:,0:self.max_seq_len]; temp[:, self.max_seq_len] = input_ids[:, -1]
                input_ids = temp
                attention_mask = attention_mask[:,0:self.max_seq_len]
            if truncate == 'first':
                temp = input_ids[:,-(self.max_seq_len):]; temp[:, 0] = input_ids[:, 0]
                input_ids = temp
                attention_mask = attention_mask[:,-(self.max_seq_len):]
            cur_max_seq_len = max(torch.sum(attention_mask, dim=-1))
        if truncate: assert cur_max_seq_len <= self.max_seq_len
        return input_ids, attention_mask

    def refine_tokenizer(self, tokenizer):
        for token in self.tokens_add:
            tokenizer.add_tokens(token)
        return tokenizer

    def get_vector(self, tokenizer, truncate='tail', only=None):
        self.tokenizer = tokenizer
        speaker_fn, label_fn = self.tokenizer_['speakers']['stoi'], self.tokenizer_['labels']['ltoi']
        for desc, dialogs in self.datas['text'].items():
            if only is not None and desc!=only: continue
            dialogs_embed = []
            for dialog in dialogs:
                embedding = tokenizer(dialog['texts'], max_length=self.max_seq_len, padding='max_length', return_tensors='pt')
                input_ids, attention_mask = self.vector_truncate(embedding, truncate=truncate)
                speakers = [speaker_fn[speaker] for speaker in dialog['speakers']]
                labels = [label_fn[label] for label in dialog['labels']]
                dialog_embed = {
                    'index': dialog['index'],
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'speakers': torch.tensor(speakers),
                    'labels': torch.tensor(labels),
                }
                dialogs_embed.append(dialog_embed)

            self.info['total_samples_num'][desc] = len(dialogs_embed)
            self.info['max_token_num'][desc] = max([sample['attention_mask'].shape[-1] for sample in dialogs_embed])
            self.datas['vector'][desc] = dialogs_embed

    def get_dataloader(self, batch_size, shuffle=None, only=None):
        if shuffle is None:
            shuffle = {'train': True, 'valid': False, 'test': False}

        dataloader = {}
        for desc, data_embed in self.datas['vector'].items():
            if only is not None and desc!=only: continue
            dataloader[desc] = DataLoader(dataset=data_embed, batch_size=batch_size, shuffle=shuffle[desc], collate_fn=self.collate_fn)
            
        return dataloader

    def collate_fn(self, dialogs):
        max_token_num = max([max(dialog['attention_mask'].sum(dim=-1)) for dialog in dialogs])
        ## 获取 batch
        inputs = {}
        for col, pad in self.batch_cols.items():
            if 'index' in col: 
                temp = torch.tensor([dialog[col] for dialog in dialogs])
            if 'ids' in col or 'mask' in col:
                temp = pad_sequence([dialog[col] for dialog in dialogs], batch_first=True, padding_value=pad)[:,:,0:max_token_num]
            if 'speakers' in col or 'labels' in col:
                temp = pad_sequence([dialog[col] for dialog in dialogs], batch_first=True, padding_value=pad)
            inputs[col] = temp

        return inputs


class ERCDataset_Single(ERCDataset_Multi):
    def get_vector(self, args=None, tokenizer=None, method='tail', only=None):
        speaker_fn, label_fn = self.speakers['ntoi'], self.labels['ltoi']
        if args.anonymity: 
            tokenizer = self.refine_tokenizer(tokenizer) # 更新字典
            speaker_fn = self.speakers['atoi']
            
        self.args, self.tokenizer = args, tokenizer
        for desc, data in self.datas['text'].items():
            if only is not None and desc!=only: continue
            data_embed = []
            for item in data:
                embedding = tokenizer(item['text'], return_tensors='pt')
                input_ids, attention_mask = self.vector_truncate(embedding, method='first')
                speaker, label = speaker_fn[item['speaker']], label_fn[item['label']]
                item_embed = {
                    'index': item['index'],
                    'input_ids': input_ids.squeeze(dim=0),
                    'attention_mask': attention_mask.squeeze(dim=0),
                    'speaker': torch.tensor(speaker),
                    'label': torch.tensor(label),
                }
                data_embed.append(item_embed)

            self.datas['vector'][desc] = data_embed

    def collate_fn(self, samples):
        ## 获取 batch
        inputs = {}
        for col, pad in self.batch_cols.items():
            if 'ids' in col or 'mask' in col:  
                inputs[col] = pad_sequence([sample[col] for sample in samples], batch_first=True, padding_value=pad)
            else: 
                inputs[col] = torch.tensor([sample[col] for sample in samples])

        return inputs

def get_basic_dataset(args, desc='multi'):
    if os.path.exists(args.basic_data_path):
        dataset = torch.load(args.basic_data_path)
    else:
        if desc == 'single': dataset = ERCDataset_Single(args)
        else: dataset = ERCDataset_Multi(args)
        torch.save(dataset, args.basic_data_path)
    return dataset

def get_special_dataset(path, args, data_fn, tokenizer=None, truncate='tail'):
    if os.path.exists(path):
        return torch.load(path)

    ## 重新构建 Dataset
    dataset = data_fn(path=args.file['data'], lower=True)
    if tokenizer is None:
        if args.model['plm'] is not None: # PLM tokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model['plm'])
        else: # Glove tokenizer
            tokenizer = get_tokenizer(path=None, dataset=dataset)
        dataset.get_vector(tokenizer, truncate=truncate)
        torch.save(dataset, path)
        
    return dataset

if __name__ == "__main__":
    data_path = f'./datasets/erc/meld/'
    dataset = ERCDataset_Multi(data_path, lower=True)
    tokens = []
    for desc, samples in dataset.datas['text'].items():
        for sample in samples:
            for text in sample['texts']:
                tokens.extend(text.split(' '))
    tokenizer = get_tokenizer(path=data_path+'glove.tokenizer', tokens=tokens)
    dataset.get_vector(tokenizer, truncate=None)

    input()