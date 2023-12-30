import os, torch, itertools, random
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from xml.etree import ElementTree as ET

## dataset
pos_tag = {
    'PUNCT': 0,  # 标点符号: ".,()"
    'SYM': 0,    # 符号: "$, &"
    'PROPN': 0,  # 专有名词
    'DET': 0,    # 限定词: "a, an, the"
    'SPACE': 0,  # 空格

    'VERB': 1,   # 动词
    'NOUN': 1,   # 名词
    'PRON': 1,   # 代词: "you, they"
    'ADP': 1,    # 介词: "in, to, during"
    'X': 1,      # 其他
    'NUM': 1,    # 数词
    'PART': 1,   # 虚词: "not, 's"
    
    'SCONJ': 2,  # 从属连词: "if, while, that"
    'CCONJ': 2,  # 并列连词: "and, or, but"
    'AUX': 2,    # 助动词: "has, will, should"
    'INTJ': 2,   # 感叹词: "hello, ouch"

    'ADV': 3,    # 副词
    'ADJ': 3,    # 形容词
}

class ABSADataset(Dataset):
    def __init__(self, args):
        args.params_train.do_test = False
        self.name = args.task
        self.container_init() # 初始化容器信息

        # 解析数据集
        self.get_dataset(args.params_file.data) 
        self.n_class = len(set([item['polarity'] for item in self.datas['text']['train']]))

        # 文本数字化
        if args.params_model.tokenizer is None: # 加载预训练tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(args.params_model.plm)
        else: # args.params_model.tokenizer 是tokenizer的路径，需自己构建
            self.tokenizer = get_tokenizer(
                path=args.params_model.tokenizer, 
                dataset=self
            )
        self.get_vector()

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

    def get_dataset(self, path, form='seg'):
        for desc in ['train', 'test']:
            data_path = f'{path}/{desc}.seg'
            with open(data_path, 'r', encoding='utf-8') as fr:
                lines, samples = fr.readlines(), []
                for i in range(0, len(lines), 3):
                    text_l, _, text_r = [s.lower().strip() for s in lines[i].partition("$T$")]
                    aspect, polarity = lines[i + 1].lower().strip(), int(lines[i + 2].strip())+1 
                    sentence = text_l + ' ' + aspect + ' ' + text_r
                    aspect_pos = [len(text_l), len(text_l)+len(aspect)]
                    assert sentence[aspect_pos[0]+1: aspect_pos[1]+1] == aspect
                    sample = {
                        'index': i//3, 
                        'aspect': aspect,
                        'sentence': sentence,
                        'asp_pos': aspect_pos,
                        'polarity': polarity,
                    }
                    samples.append(sample)
            self.datas['text'][desc] = samples

    def get_vector(self, tokenizer=None, only=None): 
        ## 确定 tokenizer
        if tokenizer is None: tokenizer = self.tokenizer 
        else: self.tokenizer = tokenizer        
    
        ## 样本数字化
        for desc, data in self.datas['text'].items():
            if only is not None and desc!=only: continue
            data_embed = []
            for item in data:
                # embedding = tokenizer.encode_plus(item['sentence'], item['aspect'], max_length=self.max_seq_len, padding='max_length', return_tensors='pt')
                embedding = tokenizer.encode_plus(item['sentence'], item['aspect'], return_tensors='pt')
                item_embed = {
                    'index': item['index'],
                    'input_ids': embedding.input_ids.squeeze(dim=0),
                    'attention_mask': embedding.attention_mask.squeeze(dim=0),
                    'token_type_ids': embedding.token_type_ids.squeeze(dim=0),
                    'polarity': item['polarity'],
                }
                data_embed.append(item_embed)

            self.datas['vector'][desc] = data_embed

    def get_dataloader(self, batch_size, shuffle=None, collate_fn=None, only=None):
        if shuffle is None: shuffle = {'train': True, 'valid': False, 'test': False}
        if collate_fn is None: collate_fn = self.collate_fn

        for desc, data_embed in self.datas['vector'].items():
            if only is not None and desc!=only: continue
            self.datas['dataloader'][desc] = DataLoader(dataset=data_embed, batch_size=batch_size, shuffle=shuffle[desc], collate_fn=collate_fn)

        if 'valid' not in self.datas['dataloader']:
            self.datas['dataloader']['valid'] = self.datas['dataloader']['test']

    def collate_fn(self, samples):
        ## 获取 batch
        inputs = {}
        for col, pad in self.batch_cols.items():
            if 'ids' in col or 'mask' in col:  
                inputs[col] = pad_sequence([sample[col] for sample in samples], batch_first=True, padding_value=pad)
            else: 
                inputs[col] = torch.tensor([sample[col] for sample in samples])

        return inputs

def get_special_dataset(args, dataset_fn=None):
    if dataset_fn is None: dataset_fn = ABSADataset
    if os.path.exists(args.params_model.data):
        dataset = torch.load(args.params_model.data)
    else:
        dataset = dataset_fn(args)
        torch.save(dataset, args.params_model.data)
        
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
    def __init__(self, embed_dim, pad_token='[PAD]', unk_token='[UNK]', mask_token='[MASK]', lower=True):
        self.lower = lower
        self.pad_token = pad_token # '[PAD]'
        self.unk_token = unk_token # '[UNK]'
        self.mask_token = mask_token # '[MASK]'
        self.vocab = {pad_token: 0, unk_token: 1, mask_token: 2}
        self.ids_to_tokens = {0: pad_token, 1: unk_token, 2: mask_token}
        self.words = {}
        self.embed_dim = embed_dim
    
    def count(self, ele):
        if self.lower: ele = ele.lower()
        if ele not in self.words: self.words[ele] = 0
        self.words[ele] += 1

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

    def encode(self, sample, max_length=None, return_tensors='pt'):
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
        tokenizer = Tokenizer(embed_dim=300, )
        tokens = [item['sentence'].split(' ') for item in dataset.datas['text']['train']]
        tokens.extend([item['sentence'].split(' ') for item in dataset.datas['text']['test']]) # 性能好点儿
        iterSupport(tokenizer.count, tokens) # 统计词频
        tokenizer.get_vocab(min_count=0) # 获得指向
        tokenizer.get_word_embedding() # 获取embedding
        torch.save(tokenizer, path)
        
    return tokenizer

                 



