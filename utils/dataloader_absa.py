import os, torch, random, spacy, json, copy
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from xml.etree import ElementTree as ET
from utils.similarity import similarity_by_editdistance

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

    def encode(self, words, return_tensors='pt', add_special_tokens=False):
        if not isinstance(words, list): words = [words]
        input_ids = []
        for word in words:
            if word in self.vocab: input_ids.append(self.vocab[word])
            else: input_ids.append(self.vocab[self.unk_token])
        
        input_ids = torch.tensor(input_ids)
        return {
            'input_ids': input_ids,
            'attention_mask': torch.ones_like(input_ids),
        }

    def encode_(self, sample, max_length=None, return_tensors='pt', add_special_tokens=False):
        tokens_snt, tokens_asp = sample['sentence'].split(' '), sample['aspect'].split(' ')
        attention_mask = torch.tensor([1] * len(tokens_snt))
        input_ids = self.tokens_to_ids(tokens_snt)

        # 定位aspect位置
        attention_mask_asp = torch.zeros_like(attention_mask)
        char_start, char_end, char_point = sample['asp_pos'][0], sample['asp_pos'][1], 0
        for i, token in enumerate(tokens_snt):
            if char_point >= char_start and char_point <= char_end:
                if token in tokens_asp:
                    attention_mask_asp[i] = 1
            char_point += len(token)+1

        assert all(input_ids[attention_mask_asp.type(torch.bool)] == self.tokens_to_ids(tokens_asp))

        if max_length is not None: # 需要截断
            pass

        return Embedding({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'attention_mask_asp': attention_mask_asp,
        })

def get_tokenizer(path, dataset):
    ## 保存tokenizer
    if os.path.exists(path):
        tokenizer = torch.load(path)
    else:
        tokenizer = Tokenizer(embed_dim=300, lower=False)
        tokens = [item['tokens'] for item in dataset.datas['text']['train']]
        tokens.extend([item['sentence'].split(' ') for item in dataset.datas['text']['test']]) # 性能好点儿
        iterSupport(tokenizer.count, tokens) # 统计词频
        tokenizer.get_vocab(min_count=0) # 获得指向
        tokenizer.get_word_embedding() # 获取embedding
        torch.save(tokenizer, path)
        
    return tokenizer


## absa dataset
class ABSADataset_MA(Dataset):
    def __init__(self, data_path, lower=False):
        self.lower = lower
        self.name = ['absa', data_path.split('/')[-2]]
        self.container_init() # 初始化容器信息
        for desc in ['train', 'test']:
            self.datas['text'][desc] = self.get_dataset(data_path, desc)
        self.n_class = 3

    def container_init(self, only='all'):
        self.info = {
            'max_seq_token_num': {}, # 句子 最长长度
            'max_asp_token_num': {}, # aspect 最长长度
            'total_samples_num': {}, # 样本数量
            'class_category': 0, # 情感类别
        }

        # 初始化数据集要保存的内容 
        self.datas = {
            'text': {},        # 解析后纯文本
            'vector': {},   # 文本分词后数字化表示
            'dataloader': {},  # dataloader用于构建batch
        }

        # tokenizer
        self.tokenizer_ = {}

    def get_tokenizer(self, samples, names):
        if not isinstance(names, list): names = [names]
        for name in names:
            if name not in samples[0]: continue
            tokenizer = Tokenizer(is_full=False, lower=False)

            tokenizer.pad_token, tokenizer.unk_token = '<pad>', '<unk>'
            addition = [tokenizer.pad_token, tokenizer.unk_token]
            if 'deprel' in name: 
                tokenizer.self_token = '<self>'
                addition += [tokenizer.self_token]
            if 'polarity' in name: addition = []

            iterSupport(tokenizer.count, [addition]+[sample[name] for sample in samples])
            tokenizer.get_vocab(min_count=0)
            self.tokenizer_[name] = tokenizer
        
    def get_dataset(self, path, desc):
        data_path = f'{path}/{desc}.multiple.json'
        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as fr:
                samples = json.load(fr)
            return samples
        else:
            raw_data_path = f'{path}/{desc}.raw.json'
            with open(raw_data_path, 'r', encoding='utf-8') as fp:
                raw_samples, samples = json.load(fp), []
            
            nlp = spacy.load('en_core_web_trf')
            for sample in raw_samples:
                if not self.lower: tokens = sample['token']
                else: tokens = [token.lower() for token in sample['token']]                    
                assert '' not in tokens and ' ' not in tokens

                # 确保 aspect 位置正确, 且按顺序出现
                aspects = sample['aspects']
                for aspect in aspects:
                    if self.lower: aspect['term'] = [token.lower() for token in aspect['term']] 
                    assert aspect['term'] == tokens[aspect['from']: aspect['to']]
                asp_begins = [aspect['from'] for aspect in aspects]
                if not all(x<=y for x, y in zip(asp_begins, asp_begins[1:])):
                    modify_index = sorted(range(len(asp_begins)), key=lambda x:asp_begins[x])
                    aspects= [aspects[idx] for idx in modify_index]

                sentence = ' '.join(tokens)
                tags, deprels, heads, words = [], [], [], []
                for word in nlp(sentence):
                    tags.append(word.tag_)
                    deprels.append(word.dep_)
                    heads.append(word.head.i)
                    words.append(str(word))
                # head 指向 i：第i个token的父节点是第i个head对应位置的节点
                heads = [-1 if head==i else head for i, head in enumerate(heads)]
                assert -1 in heads
                
                if len(words) != len(tokens):
                    assert len(words) > len(tokens)
                    aspect_token_range, new_aspect_token_range, decay = [], [], 0
                    for aspect in aspects: aspect_token_range.extend([aspect['from'], aspect['to']])
                    for t, token in enumerate(tokens):
                        if t in aspect_token_range: new_aspect_token_range.append(t+decay)
                        if token == words[t+decay]: continue
                        if t == len(tokens)-1: continue
                        next_token, decay_add = tokens[t+1], 1
                        while words[t+decay+decay_add] != next_token: decay_add += 1
                        decay += (decay_add-1)

                    for i, val in enumerate(new_aspect_token_range):
                        if i%2 == 1: continue
                        aspects[i//2]['from'] = new_aspect_token_range[i]
                        aspects[i//2]['to'] = new_aspect_token_range[i+1]
                        if words[aspects[i//2]['from']:aspects[i//2]['to']] != aspects[i//2]['term']:
                            print(f"{words[aspects[i//2]['from']:aspects[i//2]['to']]} -> {aspects[i//2]['term']}")
                 
                samples.append({
                    'index': len(samples),
                    'aspects': aspects,
                    'sentence': sentence,
                    'tokens': words,
                    'heads': heads,
                    'deprels': deprels,
                    'tags': tags,
                })

            samples_json = json.dumps(samples, indent=4)
            with open(f'{path}{desc}.multiple.json', 'w') as fw:
                fw.write(samples_json)
            
            return samples
    
    def get_vector(self, args=None, tokenizer=None, only=None):      
        def get_embedding(sample, tokenizer):
            tokens, ids, mask_asp = sample['tokens'], [], []
            for i, token in enumerate(tokens):
                idx = tokenizer.encode(token, return_tensors='pt', add_special_tokens=False)
                ids.extend(idx)
                if i>=sample['aspect']['from'] and i<sample['aspect']['to']:
                    mask_asp.extend([1]*len(idx))
                else:
                    mask_asp.extend([0]*len(idx))
            
            assert len(ids) == len(mask_asp)
            return {
                'input_ids': torch.tensor(ids),
                'attention_mask': torch.tensor([1]*len(ids)),
                'attention_mask_asp': torch.tensor(mask_asp),
            }

        self.tokenizer = tokenizer
        for desc, samples in self.datas['text'].items():
            if only is not None and desc!=only: continue
            samples_embed = []
            for sample in samples:
                # embedding = tokenizer.encode_plus(item['sentence'], item['aspect'], max_length=self.max_seq_len, padding='max_length', return_tensors='pt')
                # embedding = tokenizer.encode_plus(item['sentence'], item['aspect'], return_tensors='pt')
                embedding = get_embedding(sample, tokenizer)
                sample_embed = {
                    'index': len(samples_embed),
                    'input_ids': embedding['input_ids'],
                    'attention_mask': embedding['attention_mask'],
                    'attention_mask_asp': embedding['attention_mask_asp'],
                    'polarity': self.tokenizer_['pols'].vocab[sample['aspect']['polarity']],
                }
                samples_embed.append(sample_embed)

            self.datas['vector'][desc] = samples_embed


        self.args, self.tokenizer = args, tokenizer
        self.mask_token, self.mask_token_id = tokenizer.mask_token, tokenizer.mask_token_id
        self.eos_token, self.eos_token_id = tokenizer.eos_token, tokenizer.eos_token_id
        for desc, data in self.datas['text'].items():
            if only is not None and desc!=only: continue
            data_embed = []
            for item in data:
                query, labels = '', []
                for aspect in item['aspects']:
                    if query != '': query += f' {self.eos_token} '
                    query += f"the sentiment of {aspect['term']} is {self.mask_token}"
                    labels.append(aspect['polarity'])

                embedding = tokenizer.encode_plus(item['text'], query, return_tensors='pt')
                item_embed = {
                    'index': item['index'],
                    'input_ids': embedding.input_ids.squeeze(dim=0),
                    'attention_mask': embedding.attention_mask.squeeze(dim=0),
                    'token_type_ids': embedding.token_type_ids.squeeze(dim=0),
                    'polarity_ids': torch.tensor(labels),
                }
                data_embed.append(item_embed)

            self.datas['vector'][desc] = data_embed

    def split_dev(self, rate=0.2, method='random'):
        samps_len = len(self.datas['text']['train'])
        valid_len = int(samps_len*rate); train_len = samps_len-valid_len

        if method == 'random':
            sel_index = list(range(samps_len)); random.shuffle(sel_index)
            train = [self.datas['text']['train'][index] for index in sel_index[0:train_len]]
            valid = [self.datas['text']['train'][index] for index in sel_index[-valid_len:]]

        for i, sample in enumerate(train): sample['index'] = i
        for i, sample in enumerate(valid): sample['index'] = i

        self.datas['text']['train'] = train
        self.datas['text']['valid'] = valid

    def get_dataloader(self, batch_size, shuffle=None, collate_fn=None, only=None, split=None):
        if shuffle is None: shuffle = {'train': True, 'valid': False, 'test': False}
        if collate_fn is None: collate_fn = self.collate_fn

        dataloader = {}
        for desc, data_embed in self.datas['vector'].items():
            if only is not None and desc!=only: continue
            dataloader[desc] = DataLoader(dataset=data_embed, batch_size=batch_size, shuffle=shuffle[desc], collate_fn=collate_fn)
            
        if only is None and 'valid' not in dataloader: dataloader['valid'] = dataloader['test']
        return dataloader

    def collate_fn(self, samples):
        ## 获取 batch
        inputs = {}
        for col, pad in self.batch_cols.items():
            if 'ids' in col or 'mask' in col:  
                inputs[col] = pad_sequence([sample[col] for sample in samples], batch_first=True, padding_value=pad)
            else: 
                inputs[col] = torch.tensor([sample[col] for sample in samples])

        return inputs

class ABSADataset(ABSADataset_MA):
    def __init__(self, data_path, lower=False):
        self.lower = lower
        self.name = ['absa', data_path.split('/')[-2]]
        self.container_init() # 初始化容器信息
        if not os.path.exists(data_path+'train.multiple.json'): ABSADataset_MA(data_path, lower=lower)
        for desc in ['train', 'test']:
            self.datas['text'][desc] = self.get_dataset(data_path, desc) # 解析数据集
        self.datas['text']['valid'] = self.datas['text']['test'] 
        self.get_tokenizer(self.datas['text']['train'], names=['deprels','tags','distance','polarity'])
        self.n_class = 3

    def get_dataset(self, path, desc):
        def get_distance(sample):
            dis = {'left':[], 'mid': [], 'right':[]}
            for i, token in enumerate(sample['tokens']):
                if i < sample['aspect']['from']: dis['left'].extend([i-sample['aspect']['from']])
                if i >= sample['aspect']['to']: dis['right'].extend([i+1-sample['aspect']['to']])
                if i>=sample['aspect']['from'] and i<sample['aspect']['to']: dis['mid'].extend([0])
            return dis['left']+dis['mid']+dis['right']

        raw_data_path = f'{path}/{desc}.multiple.json'
        if not os.path.exists(raw_data_path): return None
        with open(raw_data_path, 'r', encoding='utf-8') as fp:
            raw_samples, samples = json.load(fp), []
        for sample in raw_samples:
            aspects = sample['aspects']
            for aspect in aspects:
                temp = copy.deepcopy(sample)
                temp['index'] = len(samples)
                temp['aspect'] = ' '.join(aspect['term'])
                temp['aspect_pos'] = [aspect['from'], aspect['to']]
                if ' '.join(temp['tokens'][temp['aspect_pos'][0]:temp['aspect_pos'][1]]) != temp['aspect']:
                    print(f"{' '.join(temp['tokens'][temp['aspect_pos'][0]:temp['aspect_pos'][1]])} -> {temp['aspect']}")
                temp['polarity'] = aspect['polarity']
                #temp['distance'] = get_distance(temp)
                del temp['aspects']
                samples.append(temp)

        return samples

    def get_vector(self, tokenizer, only=None, is_dep=False): 
        def get_adjs(seq_len, sample_embed, self_token_id, directed=True, loop=False):
            # head -> adj (label: dep)
            idx_asp = [idx for idx, val in enumerate(sample_embed['attention_mask_asp']) if val]
            heads, deps = sample_embed['dep_heads'], sample_embed['dep_deprels']
            adjs = np.zeros((seq_len, seq_len), dtype=np.float32)
            edges = np.zeros((seq_len, seq_len), dtype=np.int64)
            # head 指向 idx：第i个token的父节点是第i个head对应位置的节点
            for idx, head in enumerate(heads):
                if idx in idx_asp: # 是 aspect
                    for k in idx_asp: 
                        adjs[idx, k], edges[idx, k] = 1, self_token_id
                        adjs[k, idx], edges[k, idx] = 1, self_token_id
                if head != -1: # non root
                    adjs[head, idx], edges[head, idx] = 1, deps[idx]
                if not directed: # 无向图
                    adjs[idx, head], edges[idx, head] = 1, deps[idx]
                if loop: # 自身与自身相连idx
                    adjs[idx, idx], edges[idx, idx] = 1, self_token_id
            
            sample_embed['dep_graph_adjs'] = torch.tensor(adjs)
            sample_embed['dep_graph_edges'] = torch.tensor(edges)
            return sample_embed

        def get_dependency(sample, sample_embed, tokenizers):
            sample_embed['dep_heads'] = torch.tensor(sample['heads'])
            for desc in ['deprels', 'tags']:
                temp, vocab, unk_token = sample[desc], tokenizers[desc].vocab, tokenizers[desc].unk_token
                temp_id = [vocab[unk_token] if item not in vocab else vocab[item] for item in temp]
                sample_embed['dep_'+desc] = torch.tensor(temp_id)
            
            dep_self_token_id = tokenizers['deprels'].vocab[tokenizers['deprels'].self_token]
            sample_embed = get_adjs(len(sample_embed['dep_heads']), sample_embed, dep_self_token_id)
            return sample_embed
        
        def get_embedding(sample, tokenizer):
            tokens, ids, mask_asp = sample['tokens'], [], []
            tokens_asp = tokens[sample['aspect_pos'][0]: sample['aspect_pos'][1]]
            dis_asp = {'left':[], 'mid': [], 'right':[]}
            tokens_full = [tokenizer.cls_token] + tokens + [tokenizer.sep_token] + tokens_asp + [tokenizer.sep_token]
            asp_pos_full = [sample['aspect_pos'][0]+1, sample['aspect_pos'][1]+1]
            assert tokens_full[asp_pos_full[0]:asp_pos_full[1]] == tokens_asp
            for i, token in enumerate(tokens_full):
                idx = tokenizer.encode(token, return_tensors='pt', add_special_tokens=False)[0]
                ids.extend(idx)
                # context-aspect distance
                if i < asp_pos_full[0]: dis_asp['left'].extend([i-asp_pos_full[0]]*len(idx))
                if i >= asp_pos_full[1]: dis_asp['right'].extend([i+1-asp_pos_full[1]]*len(idx))
                # aspect mask
                if i>=asp_pos_full[0] and i<asp_pos_full[1]:
                    dis_asp['mid'].extend([0]*len(idx))
                    mask_asp.extend([1]*len(idx))
                else:
                    mask_asp.extend([0]*len(idx))

            assert len(ids) == len(mask_asp)
            distance_asp = dis_asp['left']+dis_asp['mid']+dis_asp['right']
            return {
                'input_ids': torch.tensor(ids),
                'attention_mask': torch.tensor([1]*len(ids)),
                'attention_mask_asp': torch.tensor(mask_asp),
                'asp_dis_ids': torch.tensor([dis+100 for dis in distance_asp])
            }
                
        self.tokenizer = tokenizer
        for desc, samples in self.datas['text'].items():
            if samples is None: continue
            if only is not None and desc!=only: continue
            samples_embed = []
            for sample in samples:
                # embedding = tokenizer.encode_plus(item['sentence'], item['aspect'], max_length=self.max_seq_len, padding='max_length', return_tensors='pt')
                # embedding = tokenizer.encode_plus(item['sentence'], item['aspect'], return_tensors='pt')
                embedding = get_embedding(sample, tokenizer)
                sample_embed = {
                    'index': sample['index'],
                    'input_ids': embedding['input_ids'],
                    'attention_mask': embedding['attention_mask'],
                    'attention_mask_asp': embedding['attention_mask_asp'],
                    'asp_dis_ids': embedding['asp_dis_ids'],
                    'label': self.tokenizer_['polarity'].vocab[sample['polarity']],
                }
                if is_dep: sample_embed = get_dependency(sample, sample_embed, tokenizers=self.tokenizer_) # 解析句法依赖信息
                sample['label'] = sample_embed['label']
                samples_embed.append(sample_embed)

            self.datas['vector'][desc] = samples_embed

    def collate_fn(self, samples):
        ## 获取 batch
        inputs = {}
        for col, pad in self.batch_cols.items():
            if 'ids' in col or 'mask' in col:  
                inputs[col] = pad_sequence([sample[col] for sample in samples], batch_first=True, padding_value=pad)
            else: 
                inputs[col] = torch.tensor([sample[col] for sample in samples])

        return inputs


def get_special_dataset(info_data, info_tokenizer, data_fn=None, is_dep=False):
    """
    info_data: 0: processed dataset path; 1: raw dataset path
    info_tokenizer: 0: plm; 1: tokenizer saved path
    data_fn: specific Dataset
    """
    if data_fn is None: data_fn = ABSADataset
    if os.path.exists(info_data[0]):
        dataset = torch.load(info_data[0])
    else:
        dataset = data_fn(info_data[1], lower=True)
        if info_tokenizer[0] is not None:
            tokenizer = AutoTokenizer.from_pretrained(info_tokenizer[0])
        else:
            tokenizer = get_tokenizer(path=info_tokenizer[1], dataset=dataset)
        dataset.get_vector(tokenizer, is_dep=is_dep)
        torch.save(dataset, info_data[0])
        
    return dataset

def parse_xml(path):
    sentences, samples = ET.parse(path).getroot(), []
    for sentence in sentences:
        for item in sentence:
            if item.tag == 'text':
                sample = {'text': item.text, 'aspects': []}
            else:
                for asp in item:
                    term, polarity = asp.attrib['term'], asp.attrib['polarity']
                    position = [int(asp.attrib['from']), int(asp.attrib['to'])]
                    assert sample['text'][position[0]:position[1]] == term
                    sample['aspects'].append({ 'term': term, 'polarity': polarity, 'position': position })
                
                asp_begins = [aspect['position'][0] for aspect in sample['aspects']]
                if not all(x<=y for x, y in zip(asp_begins, asp_begins[1:])):
                    modify_index = sorted(range(len(asp_begins)), key=lambda x:asp_begins[x])
                    sample['aspects'] = [sample['aspects'][idx] for idx in modify_index]

                samples.append(sample) # 有aspect才存储

    return samples

def change_aspects(dataset, glove_tokenizer, method='null'):
    aspects = [aspect for aspect in dataset.info['asp_lab_idx']]
    for desc, samples in dataset.datas['text'].items():
        if desc != 'test': continue
        new_samples = []
        for sample in samples:
            temp = copy.deepcopy(sample)

            # aspect 替换为 null
            if method == 'null': 
                a_aspect = ['null'] 
            # aspect 替换为相近的 ample 类型的
            if method == 'ample':
                rank_aspects = similarity_by_editdistance(temp['aspect'], aspects)
                for asp in rank_aspects:
                    rets = dataset.info['asp_lab_idx'][asp]
                    if len(rets[0]) and len(rets[1]) and len(rets[2]):
                        a_aspect = asp.split(' ')
                        break
            # aspect 替换为相近的 few 类型的
            if method == 'few': 
                rank_aspects = similarity_by_editdistance(temp['aspect'], aspects)
                for asp in rank_aspects:
                    rets = dataset.info['asp_lab_idx'][asp]
                    if len(rets[0])+len(rets[1])+len(rets[2]) <= 2 and \
                        len(rets[0])+len(rets[1])+len(rets[2]) > 0:
                        a_aspect = asp.split(' ')
                        break

            tokens, new_tokens = temp['tokens'], []
            aspect_pos_list = list(range(temp['aspect_pos'][0], temp['aspect_pos'][1]))
            for ti, token in enumerate(tokens):
                if ti == aspect_pos_list[0]:
                    new_tokens.extend(a_aspect)
                    temp['aspect_pos'] = [ti, ti+len(a_aspect)]
                elif ti not in aspect_pos_list:
                    new_tokens.append(token)
            temp['aspect'] = ' '.join(a_aspect)
            temp['tokens'] = new_tokens
            temp['sentence'] = ' '.join(new_tokens)
            new_samples.append(temp)
        
    dataset.datas['text'][f'test_{method}'] = new_samples
    dataset.get_vector(dataset.tokenizer, only=f'test_{method}')
    dataset.shuffle[f'test_{method}'] = False
    return dataset
            