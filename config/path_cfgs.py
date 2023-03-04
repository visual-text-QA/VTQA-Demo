import os


class PATH:
    def __init__(self):

        # vqav2 dataset root path
        self.DATASET_PATH = '../data/'

        self.init_path()

    def init_path(self):

        self.PRED_PATH = './results/pred/'
        self.LOG_PATH = './results/log/'
        self.CKPTS_PATH = './results/ckpts/'

        if 'pred' not in os.listdir('./results'):
            os.mkdir('./results/pred')

        if 'log' not in os.listdir('./results'):
            os.mkdir('./results/log')

        if 'ckpts' not in os.listdir('./results'):
            os.mkdir('./results/ckpts')

    def check_path(self):
        print('Checking dataset ...')
        print('Finished')
        print('')
