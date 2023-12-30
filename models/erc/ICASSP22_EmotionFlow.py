import torch, os, fitlog, json, random
import torch.nn as nn
from typing import List, Optional
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer, AdamW
from transformers import get_cosine_schedule_with_warmup
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
        self.n_class = len(self.labels['ltoi']) 

    def add_prompt(self, text, speaker, contexts, add_c=1):
        context_num = self.context_num
        eos_token, mask_token = self.tokenizer.eos_token, self.tokenizer.mask_token
        speaker_text = speaker + ': ' + text
        if add_c: contexts.append(speaker_text) # 记录在 context 中

        query_prompt = f'Now {speaker} feels {mask_token}'
        sample_context = f' {eos_token} '.join(contexts[-context_num:]) # 上下文，保留前8句
        final_text = sample_context + f' {eos_token} ' + query_prompt
        
        return final_text, contexts
    
    def prompt(self, dialog):
        contexts = []
        for ui in range(len(dialog['texts'])):
            dialog['texts'][ui], contexts = self.add_prompt(
                text=dialog['texts'][ui], 
                speaker=dialog['speakers'][ui], 
                contexts=contexts
            )
        return dialog

    def get_vector(self, tokenizer, truncate='tail', only=None):
        self.tokenizer = tokenizer
        speaker_fn, label_fn = self.speakers['stoi'], self.labels['ltoi']
        for desc, dialogs in self.datas['text'].items():
            if only is not None and desc!=only: continue
            dialogs_embed = []
            for dialog in dialogs:
                dialog = self.prompt(dialog)
                embedding = tokenizer(dialog['texts'], padding=True, return_tensors='pt')
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

            self.datas['vector'][desc] = dialogs_embed

def config_for_model(args, scale='base'):
    scale = args.params_model.scale
    args.params_model.about = 'emotionflow'
    args.params_model.plm = f'roberta-{scale}'

    args.params_model.data = args.params_file.data + f'dataset.{args.params_model.name[0]}.' + scale
    args.params_model.baseline = baselines[scale][args.task[-1]]

    args.params_model.tokenizer = None
    args.params_model.optim_sched = ['AdamW_', 'cosine']

    args.params_model.use_cl = 0  # 是否使用课程学习
    args.params_model.use_scl = 1 # 是否使用对比学习
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
        'input_ids': dataset.tokenizer.pad_token_id, 
        'attention_mask': 0, 
        'labels': -1, 
    }
    dataset.shuffle = {'train': True, 'valid': False, 'test': False}
    args.dataset = dataset

    model = CRFModel(
        args=args,
        n_class=dataset.n_class,
        plm=args.params_model.plm,
        )

    return model


class CRFModel(nn.Module):
    def __init__(self, args, n_class, plm=None):
        super().__init__()
        self.n_class = n_class
        self.params = args.params_model
        self.mask_token_id = args.dataset.tokenizer.mask_token_id

        self.plm_model = AutoModel.from_pretrained(plm, local_files_only=False)
        self.hidden_dim = self.plm_model.embeddings.word_embeddings.embedding_dim
        self.linear = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.clasifier = nn.Linear(self.hidden_dim, args.dataset.n_class)
        self.loss_ce = CrossEntropyLoss(ignore_index=-1)

        self.crf = CRF(n_class) # 条件随机场

    def optimizer_(self, params, iter_total=None):
        lr, lr_pre = params.learning_rate, params.learning_rate_pre
        weight_decay, adam_epsilon, warmup_ratio = params.weight_decay, params.adam_epsilon, params.warmup_ratio
        
        no_decay = ['bias', 'LayerNorm.weight']
        lr, lr_pre, weight_decay = 1e-3, 1e-5, 0.01
        model_params, warmup_params = [], []
        plm_params, crf_params = list(map(id, self.plm_model.parameters())), list(map(id, self.crf.parameters()))
        for name, model_param in self.named_parameters():
            weight_decay_ = 0 if any(nd in name for nd in no_decay) else weight_decay 
            lr_ = lr_pre if id(model_param) in plm_params else lr
            if id(model_param) in crf_params: lr_ = lr * 10

            model_params.append({'params': model_param, 'lr': lr_, 'weight_decay': weight_decay_})
            warmup_params.append({'params': model_param, 'lr': lr_/4 if id(model_param) in plm_params else lr_, 'weight_decay': weight_decay_})
            
        model_params = sorted(model_params, key=lambda x: x['lr'])
        optimizer = AdamW(model_params)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        # scheduler = get_cosine_schedule_with_warmup(
        #     optimizer, 
        #     num_warmup_steps=warmup_ratio*iter_total, 
        #     num_training_steps=iter_total
        #     )
        
        return optimizer, scheduler

    def encode(self, inputs):
        features = []
        for i in range(len(inputs['labels'])): 
            out_plm = self.plm_model(
                input_ids=inputs['input_ids'][i],
                attention_mask=inputs['attention_mask'][i],
                output_hidden_states=True,
                return_dict=True
            )
            hidden_state = out_plm.last_hidden_state
            # 取出每个样本 <mask> 位置的表示
            mask_positions = (inputs['input_ids'][i]==self.mask_token_id).max(dim=1)[1]
            mask_representations = hidden_state[torch.arange(len(mask_positions)), mask_positions, :]
            # 经过线性层后即可得到样本表示向量
            features.append(self.linear(mask_representations))

        return torch.stack(features)

    def forward(self, inputs, stage='train', loss=0):
        features, labels = self.encode(inputs), inputs['labels'] # 计算样本表示

        ## 计算交叉熵损失
        utt_mask = inputs['attention_mask'].sum(-1) >0 # 有效的utterance
        logits = self.clasifier(features)
        preds = torch.argmax(logits, dim=-1)[utt_mask].cpu()
        loss = self.loss_ce(logits[utt_mask], labels[utt_mask])

        if stage == 'train':
            crf_emissions = logits.transpose(0, 1)
            crf_utt_mask = utt_mask.transpose(0, 1)
            crf_labels = labels.transpose(0, 1)
            loss_crf = -self.crf(crf_emissions, crf_labels, mask=crf_utt_mask)/labels.shape[0]

            loss += loss_crf

        return {
            'loss': loss,
            'labels': labels[utt_mask],
            'logits': logits[utt_mask],
            'preds': preds,
        }

class CRF(nn.Module):
    """Conditional random field.

    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.

    Args:
        num_tags: Number of tags.
        batch_first: Whether the first dimension corresponds to the size of a minibatch.

    Attributes:
        start_transitions (`~torch.nn.Parameter`): Start transition score tensor of size
            ``(num_tags,)``.
        end_transitions (`~torch.nn.Parameter`): End transition score tensor of size
            ``(num_tags,)``.
        transitions (`~torch.nn.Parameter`): Transition score tensor of size
            ``(num_tags, num_tags)``.


    .. [LMP01] Lafferty, J., McCallum, A., Pereira, F. (2001).
       "Conditional random fields: Probabilistic models for segmenting and
       labeling sequence data". *Proc. 18th International Conf. on Machine
       Learning*. Morgan Kaufmann. pp. 282–289.

    .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
    """

    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.global_transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the transition parameters.

        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.global_transitions, -0.1, 0.1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(
            self,
            emissions: torch.Tensor,
            tags: torch.LongTensor,
            mask: Optional[torch.ByteTensor] = None,
            reduction: str = 'sum',
    ) -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.

        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        self._validate(emissions, tags=tags, mask=mask)
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = numerator - denominator

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        assert reduction == 'token_mean'
        return llh.sum() / mask.float().sum()

    def decode(
        self, emissions: torch.Tensor,
        mask: Optional[torch.ByteTensor] = None) -> List[List[int]]:
        """Find the most likely tag sequence using Viterbi algorithm.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.

        Returns:
            List of list containing the best tag sequence for each batch.
        """
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)

    def _validate(
            self,
            emissions: torch.Tensor,
            tags: Optional[torch.LongTensor] = None,
            mask: Optional[torch.ByteTensor] = None) -> None:
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}')

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(
            self,
            emissions: torch.Tensor,
            tags: torch.LongTensor,
            mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # speakers : (seq_length, batch_size)
        # last_turns: (seq_length, batch_size) last turn for the current speaker
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all()

        seq_length, batch_size = tags.shape
        mask = mask.float()

        # Start transition score and first emission
        # shape: (batch_size,)
        # st_transitions = torch.softmax(self.start_transitions, -1)
        # ed_transitions = torch.softmax(self.end_transitions, -1)
        # transitions = torch.softmax(self.transitions, -1)
        # emissions = torch.softmax(emissions, -1)
        # personal_transitions = torch.softmax(self.personal_transitions, -1)
        st_transitions = self.start_transitions
        ed_transitions = self.end_transitions
        score = st_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]
        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            global_transitions = self.global_transitions[tags[i - 1], tags[i]]
            score += global_transitions * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]
       
        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += ed_transitions[last_tags]

        return score

    def _compute_normalizer(
            self, emissions: torch.Tensor,
            mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length = emissions.size(0)
        batch_size = emissions.size(1)

        st_transitions = self.start_transitions
        ed_transitions = self.end_transitions
        score = st_transitions + emissions[0]
        scores = []
        scores.append(score)
        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            global_transitions = self.global_transitions
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + global_transitions + broadcast_emissions

            next_score = torch.logsumexp(next_score, dim=1)

            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            
            scores.append(score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += ed_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(
        self, emissions: torch.FloatTensor,
        mask: torch.ByteTensor) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        st_transitions = self.start_transitions
        ed_transitions = self.end_transitions
        score = st_transitions + emissions[0]
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        scores = []
        scores.append(score)
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            global_transitions = self.global_transitions

            next_score = broadcast_score + global_transitions + broadcast_emissions

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            scores.append(score)
            history.append(indices)

        # End transition score
        # shape: (batch_size, num_tags)
        score += ed_transitions

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list