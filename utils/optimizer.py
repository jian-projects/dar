import torch
from transformers import (
    AdamW,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

def optimizers(model, params, iter_total, methods=None):
    if methods is None: methods = params.optim_sched
    lr, lr_pre = params.learning_rate, params.learning_rate_pre
    weight_decay, adam_epsilon, warmup_ratio = params.weight_decay, params.adam_epsilon, params.warmup_ratio

    no_decay = ['bias', 'LayerNorm.weight']
    if 'AdamW_' in methods:
        if 'fw' in methods: plm_params = list(map(id, model.model.plm_model.parameters()))
        else: plm_params = list(map(id, model.plm_model.parameters()))
        
        #lr, lr_pre, weight_decay = 1e-3, 1e-5, 0.01
        model_params, warmup_params = [], []
        for name, model_param in model.named_parameters():
            weight_decay_ = 0 if any(nd in name for nd in no_decay) else weight_decay 
            lr_ = lr_pre if id(model_param) in plm_params else lr

            model_params.append({'params': model_param, 'lr': lr_, 'weight_decay': weight_decay_})
            warmup_params.append({'params': model_param, 'lr': lr_/4 if id(model_param) in plm_params else lr_, 'weight_decay': weight_decay_})
        
        model_params = sorted(model_params, key=lambda x: x['lr'])
        optimizer = AdamW(model_params)
        if 'linear' in methods:
            scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=iter_total*warmup_ratio, 
                num_training_steps=iter_total
                )
        if 'cosine' in methods:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=warmup_ratio*iter_total, 
                num_training_steps=iter_total
                )
        if 'none' in methods:
            scheduler = None

    if 'AdamW' in methods:
        model_params = [
            {"params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0},
        ]
        optimizer = AdamW(model_params, lr=lr_pre, eps=adam_epsilon)
        if 'linear' in methods:
            scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=iter_total*warmup_ratio,
                num_training_steps=iter_total
                )
        if 'cosine' in methods:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=warmup_ratio*iter_total, 
                num_training_steps=iter_total
                )
        if 'none' in methods:
            scheduler = None

    if 'Adam' in methods:
        l2reg, warmup_ratio = params.l2reg, params.warmup_ratio
        model_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(model_params, lr=lr, weight_decay=l2reg)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=iter_total*warmup_ratio, 
            num_training_steps=iter_total
        )
        scheduler = None

    if 'SGD' in methods:
        l2reg, warmup_ratio = params.l2reg, params.warmup_ratio
        model_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.SGD(model_params, lr=lr, weight_decay=l2reg)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=iter_total*warmup_ratio, 
            num_training_steps=iter_total
        )
        
    return optimizer, scheduler