import warnings, os, random, torch
warnings.filterwarnings("ignore")

from config import config
from utils.writer import JsonFile
from utils.processor import Processor
from utils.processor_utils import *


def get_model(args, model_name=None):
    if model_name is None: 
        model_name = [args.model['name'], args.model['backbone']]

    ## 框架模型
    if model_name[-1] is not None: 
        return None

    ## 非框架模型
    if 'dar' in model_name: from models.erc.Our_DAR import import_model

    model, dataset = import_model(args)
    init_weight(model)

    return model, dataset

def run(args):
    args = random_parameters(args)
    model, dataset = get_model(args)
    if torch.cuda.device_count() > 1: # 指定卡/多卡 训练
        model = torch.nn.DataParallel(model, device_ids=args.train['device_ids'])

    dataset.metrics = ['accuracy']
    dataset.lab_range = list(range(dataset.n_class)) if dataset.name[-1]!='ddg' else list(range(1, dataset.n_class))
    processor = Processor(args, model, dataset)
    result = processor._train()

    ## 2. 输出统计结果
    record = {
        'params': {
            'e':       args.train['epochs'],
            'es':      args.train['early_stop'],
            'lr':      args.train['learning_rate'],
            'lr_pre':  args.train['learning_rate_pre'],
            'bz':      args.train['batch_size'],
            'dr':      args.model['drop_rate'],
            'seed':    args.train['seed'],
        },
        'metric': {
            'stop':    result['epoch'],
            'tv_mf1':  result['valid']['accuracy'],
            'te_mf1':  result['test']['accuracy'],
        },
    }
    return record


if __name__ == '__main__':

    """
    tasks: 
        ddg: DailyDialog
        emn: EmoryNLP
        iec: IEMOCAP
        meld: MELD

    models: 
        dar: Audio (SER)
    """
    args = config(tasks=['erc','iec'], models=['dar', None])

    ## Parameters Settings
    args.model['scale'] = 'base'
    args.train['device_ids'] = [0]
    
    args.train['epochs'] = 15
    args.train['early_stop'] = 5
    args.train['batch_size'] = 8
    args.train['save_model'] = False
    args.train['log_step_rate'] = 1.0
    args.train['learning_rate'] = 5e-5
    args.train['learning_rate_pre'] = 5e-5

    args.model['drop_rate'] = 0.3
    
    seeds = []
    ## Cycle Training
    if seeds: # 按指定 seed 执行
        recoed_path = f"{args.file['record']}{args.model['name']}_best.json"
        record_show = JsonFile(recoed_path, mode_w='a', delete=True)
        for seed in seeds:
            args.train['seed'] = seed
            args.train['seed_change'] = False
            record = run(args)
            record_show.write(record, space=False) 
    else: # 随机 seed 执行       
        recoed_path = f"{args.file['record']}{args.model['name']}_search.json"
        record_show = JsonFile(recoed_path, mode_w='a', delete=True)
        for c in range(100):
            args.train['seed'] = random.randint(1000,9999)+c
            args.train['seed_change'] = False
            record = run(args)
            record_show.write(record, space=False)        