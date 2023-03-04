import glob
import json
import torch

import numpy as np
import torch.utils.data as Data


class DataSet(Data.Dataset):
    def __init__(self, __C, mode='train'):
        self.max_context_token = __C.MAX_CONTEXT_TOKEN
        self.max_question_token = __C.MAX_QUESTION_TOKEN
        self.pretrained_emb = np.load(__C.DATASET_PATH + 'annotations/embedding.npy')
        self.ans2id = json.load(
            open(__C.DATASET_PATH + 'annotations/ans2id.json', 'r')
        )
        self.id2ans = {str(v): k for k, v in self.ans2id.items()}
        self.token2id = json.load(
            open(__C.DATASET_PATH + 'annotations/token2id.json', 'r')
        )

        self.mode = mode

        if mode == 'train':
            self.data = json.load(
                open(__C.DATASET_PATH + 'annotations/train_cws.json', 'r')
            )
        elif mode == 'val':
            self.data = json.load(
                open(__C.DATASET_PATH + 'annotations/val_cws.json', 'r')
            )
        elif mode == 'test':
            self.data = json.load(
                open(__C.DATASET_PATH + 'annotations/test_cws.json', 'r')
            )
        elif mode == 'test_dev':
            self.data = json.load(
                open(__C.DATASET_PATH + 'annotations/test_dev_cws.json', 'r')
            )
        else:
            assert False, 'mode not right: {}'.format(mode)

        if __C.FEATURE_TYPE == 'image':
            suffix = '.jpg'
        elif __C.FEATURE_TYPE == 'grid':
            self.img_max_token = 608
            suffix = '.npy'
        else:
            self.img_max_token = 100
            suffix = '.npz'
        img_feat_path_list = glob.glob(__C.DATASET_PATH + 'images/{}/*'.format(mode) + suffix)
        self.iid_to_img_path = {
            str(int(p.split('/')[-1].split('_')[-1].split('.')[0])): p for p in img_feat_path_list
        }

        self.token_size = len(self.token2id)
        self.ans_size = len(self.ans2id)

        self.data_size = len(self.data)

        self.__C = __C

    def __getitem__(self, idx):
        img_id = int(
            self.data[idx]['image_local_path']
            .split('/')[-1]
            .split('_')[-1]
            .split('.')[0]
        )
        img_feat = self.proc_img(str(img_id), self.img_max_token)

        context = self.data[idx]['text_cws']
        context_iter = self.proc_context(context, self.max_context_token)

        question = self.data[idx]['question_cws']
        question_iter = self.proc_ques(question, self.max_question_token)

        if 'test' not in self.mode:
            answer = self.data[idx]['answer']
            answer_iter = self.proc_answer(answer)

            return (
                torch.from_numpy(img_feat),
                torch.from_numpy(context_iter),
                torch.from_numpy(question_iter),
                torch.from_numpy(answer_iter),
            )
        else:
            return (
                torch.from_numpy(img_feat),
                torch.from_numpy(context_iter),
                torch.from_numpy(question_iter),
            )

    def __len__(self):
        return self.data_size

    def proc_img(self, img_id, img_feat_pad_size):
        path = self.iid_to_img_path[img_id]
        # img_feat = np.load(path)['x'].transpose((1, 0))
        img_feat = (
            np.load(path).transpose((1, 0))
            if self.__C.FEATURE_TYPE == 'grid'
            else np.load(path)['x'].transpose((1, 0))
        )
        if img_feat.shape[0] > img_feat_pad_size:
            img_feat = img_feat[:img_feat_pad_size]

        img_feat = np.pad(
            img_feat,
            ((0, img_feat_pad_size - img_feat.shape[0]), (0, 0)),
            mode='constant',
            constant_values=0,
        )
        return img_feat

    def proc_context(self, context, max_token):
        context_id = np.zeros(max_token, np.int64)
        ix = 0
        for sent in context:
            for word in sent:
                if word in self.token2id:
                    context_id[ix] = self.token2id[word]
                else:
                    context_id[ix] = self.token2id['UNK']

                if ix + 1 == max_token:
                    break
                ix += 1

        return context_id

    def proc_ques(self, question, max_token):
        question_id = np.zeros(max_token, np.int64)
        ix = 0
        for word in question:
            if word in self.token2id:
                question_id[ix] = self.token2id[word]
            else:
                question_id[ix] = self.token2id['UNK']

            if ix + 1 == max_token:
                break
            ix += 1

        return question_id

    def proc_answer(self, answer):
        ans_score = np.zeros(self.ans2id.__len__(), np.float32)
        if answer in self.ans2id:
            ans_score[self.ans2id[answer]] = 1
        else:
            pass
        return ans_score


if __name__ == "__main__":
    from config import Cfgs

    dataset = DataSet(Cfgs())
    d = dataset[12]
