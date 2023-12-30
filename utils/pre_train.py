import copy
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
)

def mlm_dataset(self):
    p2m = self.tokenizer_['labels']['p2m']
    tokenizer = self.tokenizer
    sep_token, mask_token = tokenizer.sep_token, tokenizer.mask_token
    mask_token_id = tokenizer.mask_token_id
    for stage, dialogs in self.datas['text'].items():
        samples = []
        for dialog in dialogs:
            dialog['labels_'] = copy.deepcopy(dialog['labels'])
            for ui in range(len(dialog['texts'])): 
                if dialog['labels_'][ui] is not None:
                    dialog['labels_'][ui] = self.tokenizer(p2m[dialog['labels'][ui]]).input_ids[1:-1]
                text = f' {sep_token} '.join(dialog['texts_'][:ui+1])
                input_ids = self.tokenizer(text).input_ids
                assert sum([val== mask_token_id for val in input_ids]) == ui+1

                new_input_ids, t = [], 0
                for val in input_ids:
                    if val == mask_token_id:
                        if dialog['labels_'][t] is not None:
                            new_input_ids.extend(dialog['labels_'][t])
                        t += 1
                    else: new_input_ids.append(val)
                
                sample = {
                    'index': len(samples),
                    'input_ids': new_input_ids,
                    'attention': 1,
                    'label': 2,
                }
                samples.append(sample)

        self.datas['vector'][stage] = samples


class PreTrain():
    def __init__(self, args) -> None:
        self.dataset = args.dataset
        self.dataloader = self.get_dataloader()

        config = AutoConfig.from_pretrained(args.params_model.plm)
        model = AutoModelForMaskedLM.from_pretrained(
            args.params_model.plm,
            from_tf=bool(".ckpt" in args.params_model.plm),
            config=config,
        )

    def get_dataloader(self):
        dataset = self.dataset
        # 重构数据形式
        dataset.modify_mlm()


