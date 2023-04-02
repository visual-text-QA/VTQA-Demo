import argparse

import torch
import yaml

from config import Cfgs
from dataset import DataSet
from run import Trainer


def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--DATASET_PATH', type=str, default='../data/')

    parser.add_argument('--RESULT_PATH', type=str, default='../test_pred.json')

    parser.add_argument(
        '--MODEL_PATH', type=str, default='./results/ckpts/ckpt_demo/epoch13.pkl'
    )

    parser.add_argument(
        '--MODEL', choices=['small', 'large'], type=str, default='small'
    )

    parser.add_argument(
        '--GPU', dest='GPU', help="gpu select, eg.'0, 1, 2'", type=str, default='0'
    )

    parser.add_argument(
        '--FEATURE_TYPE', choices=['region', 'grid'], type=str, default='region'
    )

    parser.add_argument('--LANG', choices=['zh', 'en'], type=str, default='zh')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    __C = Cfgs()

    args = parse_args()
    args_dict = __C.parse_to_dict(args)

    cfg_file = "config/{}_model.yml".format(args.MODEL)
    model_path = args.MODEL_PATH
    with open(cfg_file, 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)

    args_dict = {**yaml_dict, **args_dict}
    __C.add_args(args_dict)
    __C.proc()

    print('Hyper Parameters:')
    print(__C)

    __C.check_path()

    Trainer = Trainer(__C)
    test_dataset = DataSet(__C, mode='test')
    state_dict = torch.load(model_path)['state_dict']
    Trainer.test(
        test_dataset,
        state_dict=state_dict,
        save_file_path=args.RESULT_PATH,
        mode='test',
    )
