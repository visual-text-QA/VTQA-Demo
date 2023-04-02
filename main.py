import argparse

import yaml

from config import Cfgs
from run import Trainer


def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--RUN',
        dest='RUN_MODE',
        choices=['train', 'val', 'test', 'test_dev'],
        help='{train, val, test}',
        type=str,
        required=True,
    )

    parser.add_argument(
        '--MODEL',
        dest='MODEL',
        choices=['small', 'large'],
        help='{small, large}',
        default='small',
        type=str,
    )

    parser.add_argument(
        '--EVAL_EE',
        dest='EVAL_EVERY_EPOCH',
        help='set True to evaluate the '
        'val split when an epoch finished'
        "(only work when train with "
        "'train' split)",
        type=bool,
    )

    parser.add_argument(
        '--SAVE_PRED',
        dest='TEST_SAVE_PRED',
        help='set True to save the ' 'prediction vectors' '(only work in testing)',
        type=bool,
    )

    parser.add_argument(
        '--BS', dest='BATCH_SIZE', help='batch size during training', type=int
    )

    parser.add_argument(
        '--MAX_EPOCH', dest='MAX_EPOCH', help='max training epoch', type=int
    )

    parser.add_argument(
        '--PRELOAD',
        dest='PRELOAD',
        help='pre-load the features into memory' 'to increase the I/O speed',
        type=bool,
    )

    parser.add_argument('--GPU', dest='GPU', help="gpu select, eg.'0, 1, 2'", type=str)

    parser.add_argument('--SEED', dest='SEED', help='fix random seed', type=int)

    parser.add_argument(
        '--VERSION', dest='VERSION', help='version control', default='demo', type=str
    )

    parser.add_argument('--RESUME', dest='RESUME', help='resume training', type=bool)

    parser.add_argument(
        '--CKPT_V', dest='CKPT_VERSION', help='checkpoint version', type=str
    )

    parser.add_argument(
        '--CKPT_E', dest='CKPT_EPOCH', help='checkpoint epoch', type=int
    )

    parser.add_argument(
        '--CKPT_PATH',
        dest='CKPT_PATH',
        help='load checkpoint path, we '
        'recommend that you use '
        'CKPT_VERSION and CKPT_EPOCH '
        'instead',
        type=str,
    )

    parser.add_argument(
        '--ACCU', dest='GRAD_ACCU_STEPS', help='reduce gpu memory usage', type=int
    )

    parser.add_argument(
        '--NW', dest='NUM_WORKERS', help='multithreaded loading', type=int
    )

    parser.add_argument('--PINM', dest='PIN_MEM', help='use pin memory', type=bool)

    parser.add_argument('--VERB', dest='VERBOSE', help='verbose print', type=bool)

    parser.add_argument(
        '--FEATURE_TYPE',
        choices=['region', 'grid'],
        type=str,
    )

    parser.add_argument(
        '--LANG',
        choices=['zh', 'en'],
        type=str,
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    __C = Cfgs()

    args = parse_args()
    args_dict = __C.parse_to_dict(args)

    cfg_file = "config/{}_model.yml".format(args.MODEL)
    with open(cfg_file, 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)

    args_dict = {**yaml_dict, **args_dict}
    __C.add_args(args_dict)
    __C.proc()

    print('Hyper Parameters:')
    print(__C)

    __C.check_path()

    Trainer = Trainer(__C)
    Trainer.run(__C.RUN_MODE)
