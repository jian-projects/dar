import torch, sys, os
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import AutoModel, Wav2Vec2Processor, Wav2Vec2Model
from utils.processor_utils import set_rng_seed

#from utils.contraster import *

"""
| dataset     | meld  | iec   | emn   | ddg   |

| baseline    | 67.25 | 69.74 | 40.94 | 00.00 | # test
| performance | 65.11 | 00.00 | 00.00 | 00.00 |

"""
baselines = {
    'base': {'meld': 0., 'iec': 0., 'emn': 0., 'ddg': 0}, # valid
    'large': {'meld': 0., 'iec': 0., 'emn': 0., 'ddg': 0},
}

def config_for_model(args):
    scale = args.model['scale']
    args.model['plm'] = args.file['plm_dir'] + f"Audio/wav2vec2-base-960h"
    
    args.file['data'] = args.file['data_dir'] + f"{args.train['tasks'][0]}/{args.train['tasks'][1]}/"
    # args.model['data'] = f"dataset.audio.{args.model['name']}." + scale
    args.model['baseline'] = baselines[scale][args.train['tasks'][1]]

    args.model['tokenizer'] = None
    args.model['optim_sched'] = ['AdamW_', 'cosine']
    #args.model['optim_sched'] = ['AdamW_', 'linear']

    args.train['do_test'] = False

    return args
             
def import_model(args):
    ## 1. 更新参数
    args = config_for_model(args) # 添加模型参数, 获取任务数据集
    set_rng_seed(args.train['seed'])
    
    ## 2. 导入数据
    sys.path.insert(0, args.file['data'])
    from data_loader_audio import get_specific_dataset
    dataset = get_specific_dataset(args)

    ## 3. 导入模型
    model = DAR(
        args=args,
        dataset=dataset,
        plm=args.model['plm'],
    )

    return model, dataset

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

class DAR(nn.Module):
    def __init__(self, args, dataset, plm=None):
        super().__init__()
        self.args = args
        self.n_class = dataset.n_class

        self.plm_model = Wav2Vec2Model.from_pretrained(plm, local_files_only=False) 

        # self.plm_model = AutoModel.from_pretrained(plm, local_files_only=False) 
        self.plm_model.pooler_all = PoolerAll(self.plm_model.config)
        self.hidden_dim = self.plm_model.config.hidden_size   

        self.dropout = nn.Dropout(args.model['drop_rate'])
        self.classifier = nn.Linear(self.hidden_dim, self.n_class)
        self.loss_ce = CrossEntropyLoss(ignore_index=-1)

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

    def forward(self, inputs, stage='train'):
        features = self.plm_model(inputs['audio']).last_hidden_state
        logits = self.classifier(features[:,0])
        loss = self.loss_ce(logits, inputs['label'])

        return {
            'loss':   loss,
            'logits': logits,
            'preds':  torch.argmax(logits, dim=-1).cpu(),
            'labels': inputs['label'],
        }