import datetime
import json
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data

from dataset import DataSet
from model import Net
from model.optim import adjust_lr, get_optim


class Trainer:
    def __init__(self, __C):
        self.__C = __C

        print('Loading training set ........')
        self.dataset = DataSet(__C, 'train')
        self.ans_size = self.dataset.ans_size

        self.dataset_eval = None
        if __C.EVAL_EVERY_EPOCH:
            print('Loading validation set for per-epoch evaluation ........')
            self.dataset_eval = DataSet(__C, 'val')

        self.dataset_test = None
        if __C.TEST_AFTER_TRAIN:
            print('Loading test_dev set for after-train evaluation ........')
            self.dataset_test = DataSet(__C, 'test_dev')

    def train(self, dataset, dataset_eval=None, dataset_test=None):

        # Obtain needed information
        data_size = dataset.data_size
        token_size = dataset.token_size
        pretrained_emb = dataset.pretrained_emb

        # Define the MCAN model
        net = Net(self.__C, pretrained_emb, token_size, self.ans_size)
        net.cuda()
        net.train()

        # Define the multi-gpu training if needed
        if self.__C.N_GPU > 1:
            net = nn.DataParallel(net, device_ids=self.__C.DEVICES)

        # Define the binary cross entropy loss
        # loss_fn = torch.nn.BCELoss(size_average=False).cuda()
        loss_fn = torch.nn.BCELoss(reduction='sum').cuda()

        # Load checkpoint if resume training
        if self.__C.RESUME:
            print(' ========== Resume training')

            if self.__C.CKPT_PATH is not None:
                print(
                    'Warning: you are now using CKPT_PATH args, '
                    'CKPT_VERSION and CKPT_EPOCH will not work'
                )

                path = self.__C.CKPT_PATH
            else:
                path = (
                    self.__C.CKPTS_PATH
                    + 'ckpt_'
                    + self.__C.CKPT_VERSION
                    + '/epoch'
                    + str(self.__C.CKPT_EPOCH)
                    + '.pkl'
                )

            # Load the network parameters
            print('Loading ckpt {}'.format(path))
            ckpt = torch.load(path)
            print('Finish!')
            net.load_state_dict(ckpt['state_dict'])

            # Load the optimizer paramters
            optim = get_optim(self.__C, net, data_size, ckpt['lr_base'])
            optim._step = int(data_size / self.__C.BATCH_SIZE * self.__C.CKPT_EPOCH)
            optim.optimizer.load_state_dict(ckpt['optimizer'])

            start_epoch = self.__C.CKPT_EPOCH

        else:
            if ('ckpt_' + self.__C.VERSION) in os.listdir(self.__C.CKPTS_PATH):
                shutil.rmtree(self.__C.CKPTS_PATH + 'ckpt_' + self.__C.VERSION)

            os.mkdir(self.__C.CKPTS_PATH + 'ckpt_' + self.__C.VERSION)

            optim = get_optim(self.__C, net, data_size)
            start_epoch = 0

        loss_sum = 0
        named_params = list(net.named_parameters())
        grad_norm = np.zeros(len(named_params))

        # Define multi-thread dataloader
        if self.__C.SHUFFLE_MODE in ['external']:
            dataloader = Data.DataLoader(
                dataset,
                batch_size=self.__C.BATCH_SIZE,
                shuffle=False,
                num_workers=self.__C.NUM_WORKERS,
                pin_memory=self.__C.PIN_MEM,
                drop_last=True,
            )
        else:
            dataloader = Data.DataLoader(
                dataset,
                batch_size=self.__C.BATCH_SIZE,
                shuffle=True,
                num_workers=self.__C.NUM_WORKERS,
                pin_memory=self.__C.PIN_MEM,
                drop_last=True,
            )

        # Training script
        for epoch in range(start_epoch, self.__C.MAX_EPOCH):

            # Save log information
            logfile = open(
                self.__C.LOG_PATH + 'log_run_' + self.__C.VERSION + '.txt', 'a+'
            )
            logfile.write(
                'nowTime: '
                + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                + '\n'
            )
            logfile.close()

            # Learning Rate Decay
            if epoch in self.__C.LR_DECAY_LIST:
                adjust_lr(optim, self.__C.LR_DECAY_R)

            time_start = time.time()
            # Iteration
            for step, d in enumerate(dataloader):

                optim.zero_grad()

                (image_feat, context, question, answer_idx) = (v.cuda() for v in d)

                pred = net(image_feat, context, question)

                loss = loss_fn(pred, answer_idx)
                # only mean-reduction needs be divided by grad_accu_steps
                # removing this line wouldn't change our results because the speciality of Adam optimizer,
                # but would be necessary if you use SGD optimizer.
                # loss /= self.__C.GRAD_ACCU_STEPS
                loss.backward()
                loss_sum += loss.cpu().data.numpy() * self.__C.GRAD_ACCU_STEPS

                if self.__C.VERBOSE:

                    mode_str = 'train' + '->' + 'test'

                    print(
                        "\r[version %s][epoch %2d][step %4d/%4d][%s] loss: %.4f, lr: %.2e"
                        % (
                            self.__C.VERSION,
                            epoch + 1,
                            step,
                            int(data_size / self.__C.BATCH_SIZE),
                            mode_str,
                            loss.cpu().data.numpy() / self.__C.SUB_BATCH_SIZE,
                            optim._rate,
                        ),
                        end='          ',
                    )

                # Gradient norm clipping
                if self.__C.GRAD_NORM_CLIP > 0:
                    nn.utils.clip_grad_norm_(net.parameters(), self.__C.GRAD_NORM_CLIP)

                # Save the gradient information
                for name in range(len(named_params)):
                    norm_v = (
                        torch.norm(named_params[name][1].grad).cpu().data.numpy()
                        if named_params[name][1].grad is not None
                        else 0
                    )
                    grad_norm[name] += norm_v * self.__C.GRAD_ACCU_STEPS
                    # print('Param %-3s Name %-80s Grad_Norm %-20s'%
                    #       (str(grad_wt),
                    #        params[grad_wt][0],
                    #        str(norm_v)))

                optim.step()

            time_end = time.time()
            print('Finished in {}s'.format(int(time_end - time_start)))

            # print('')
            epoch_finish = epoch + 1

            # Save checkpoint
            state = {
                'state_dict': net.state_dict(),
                'optimizer': optim.optimizer.state_dict(),
                'lr_base': optim.lr_base,
            }
            torch.save(
                state,
                self.__C.CKPTS_PATH
                + 'ckpt_'
                + self.__C.VERSION
                + '/epoch'
                + str(epoch_finish)
                + '.pkl',
            )

            # Logging
            logfile = open(
                self.__C.LOG_PATH + 'log_run_' + self.__C.VERSION + '.txt', 'a+'
            )
            logfile.write(
                'epoch = '
                + str(epoch_finish)
                + '  loss = '
                + str(loss_sum / data_size)
                + '\n'
                + 'lr = '
                + str(optim._rate)
                + '\n\n'
            )
            logfile.close()

            # Eval after every epoch
            if dataset_eval is not None:
                self.eval(dataset_eval, state_dict=net.state_dict())

            loss_sum = 0
            grad_norm = np.zeros(len(named_params))

        if self.__C.TEST_AFTER_TRAIN:
            self.test(dataset_test, state_dict=net.state_dict())

    # Evaluation
    def eval(self, dataset, state_dict=None):

        # Load parameters
        if self.__C.CKPT_PATH is not None:
            print(
                'Warning: you are now using CKPT_PATH args, '
                'CKPT_VERSION and CKPT_EPOCH will not work'
            )

            path = self.__C.CKPT_PATH
        else:
            path = (
                self.__C.CKPTS_PATH
                + 'ckpt_'
                + self.__C.CKPT_VERSION
                + '/epoch'
                + str(self.__C.CKPT_EPOCH)
                + '.pkl'
            )

        if state_dict is None:
            print('Loading ckpt {}'.format(path))
            state_dict = torch.load(path)['state_dict']
            print('Finish!')

        data_size = dataset.data_size
        token_size = dataset.token_size
        pretrained_emb = dataset.pretrained_emb

        net = Net(self.__C, pretrained_emb, token_size, self.ans_size)
        net.cuda()
        net.eval()

        if self.__C.N_GPU > 1:
            net = nn.DataParallel(net, device_ids=self.__C.DEVICES)

        net.load_state_dict(state_dict)

        dataloader = Data.DataLoader(
            dataset,
            batch_size=self.__C.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.__C.NUM_WORKERS,
            pin_memory=True,
        )

        acc = 0

        for step, d in enumerate(dataloader):
            print(
                "\r[version %s] Evaluation: [step %4d/%4d]"
                % (
                    self.__C.VERSION,
                    step,
                    int(data_size / self.__C.EVAL_BATCH_SIZE),
                ),
                end='          ',
            )

            (image_feat, context, question, answer_idx) = (v.cuda() for v in d)

            pred = net(image_feat, context, question)

            pred_np = pred.cpu().data.numpy()
            pred_argmax = np.argmax(pred_np, axis=1)
            gt = np.argmax(answer_idx.cpu().data.numpy(), axis=1)

            acc += np.sum(pred_argmax == gt)

        print('val acc: {}\n'.format(acc * 1.0 / data_size))

    def test(
        self,
        dataset,
        state_dict=None,
        save_file_path=None,
        mode='test_dev',
    ):

        # Load parameters
        if self.__C.CKPT_PATH is not None:
            print(
                'Warning: you are now using CKPT_PATH args, '
                'CKPT_VERSION and CKPT_EPOCH will not work'
            )

            path = self.__C.CKPT_PATH
        else:
            path = (
                self.__C.CKPTS_PATH
                + 'ckpt_'
                + self.__C.CKPT_VERSION
                + '/epoch'
                + str(self.__C.CKPT_EPOCH)
                + '.pkl'
            )

        if state_dict is None:
            print('Loading ckpt {}'.format(path))
            state_dict = torch.load(path)['state_dict']
            print('Finish!')
            file_name = (
                mode
                + '_result_'
                + self.__C.CKPT_VERSION
                + '_'
                + str(self.__C.CKPT_EPOCH)
                + '.json'
            )
        else:
            print('use exist state_dict')
            file_name = '{}_result_{}.json'.format(mode, self.__C.VERSION)

        data_size = dataset.data_size
        token_size = dataset.token_size
        pretrained_emb = dataset.pretrained_emb

        net = Net(self.__C, pretrained_emb, token_size, self.ans_size)
        net.cuda()
        net.eval()

        if self.__C.N_GPU > 1:
            net = nn.DataParallel(net, device_ids=self.__C.DEVICES)

        net.load_state_dict(state_dict)

        dataloader = Data.DataLoader(
            dataset,
            batch_size=self.__C.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.__C.NUM_WORKERS,
            pin_memory=True,
        )

        pred_answer_list = []

        for step, d in enumerate(dataloader):
            print(
                "\r[version %s] Evaluation: [step %4d/%4d]"
                % (
                    self.__C.VERSION,
                    step,
                    int(data_size / self.__C.EVAL_BATCH_SIZE),
                ),
                end='          ',
            )

            (image_feat, context, question) = (v.cuda() for v in d)

            pred = net(image_feat, context, question)

            pred_np = pred.cpu().data.numpy()
            pred_argmax = np.argmax(pred_np, axis=1)
            pred_answer_list += [i for i in pred_argmax]

        pred_json = []
        for p, d in zip(pred_answer_list, dataset.data):
            pred_json.append({'qid': d['qid'], 'answer': dataset.id2ans[str(p)]})

        if not save_file_path:
            save_file_path = self.__C.PRED_PATH + file_name
        json.dump(pred_json, open(save_file_path, 'w'), ensure_ascii=False)

    def run(self, run_mode):
        if run_mode == 'train':
            self.empty_log(self.__C.VERSION)
            self.train(self.dataset, self.dataset_eval, self.dataset_test)

        elif run_mode == 'val':
            self.eval(self.dataset_eval)

        elif run_mode == 'test_dev':
            self.test(self.dataset_test)

        else:
            exit(-1)

    def empty_log(self, version):
        print('Initializing log file ........')
        if os.path.exists(self.__C.LOG_PATH + 'log_run_' + version + '.txt'):
            os.remove(self.__C.LOG_PATH + 'log_run_' + version + '.txt')
        print('Finished!')
        print('')
