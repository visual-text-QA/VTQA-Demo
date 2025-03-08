from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
import torch
from PIL import Image
from transformers import (
    pipeline,
    AutoProcessor,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    ViTImageProcessor,
    LlavaPreTrainedModel,
    LlavaProcessor,
    T5Tokenizer,
)
from glob import glob
import numpy as np
from tqdm import tqdm
import pickle as pkl
import os
import json


def load_vtqa(
    split=None,
    text_processor=None,
    img_processor=None,
    max_length=None,
    max_context_token=None,
    max_question_token=None,
    max_answer_token=16,
    img_feature_type: str | None = None,
    text_question_merge=False,
    cls_target=True,
    lang: str | None = None,
    local_url=None,
    use_cws=False,
    debug=False,
    get_test_split=False,
):
    name = "all"
    if lang is not None:
        assert lang in ["en", "zh"]
        name = lang + "-" + img_feature_type if img_feature_type else lang
    elif img_feature_type is not None:
        assert img_feature_type in ["image", "region", "grid"]
        name = img_feature_type
    dataset = load_dataset(
        "CalfKing/vtqa2023",
        name,
        split=split,
        use_cws=use_cws,
        get_test_split=get_test_split,
        local_url=local_url,
        trust_remote_code=True,
    )

    token_to_ix, pretrained_emb = None, None
    if use_cws:
        cws_supp_dir = (
            dataset[0]["cws_path"] if split else dataset["train"][0]["cws_path"]
        )
        pretrained_emb = {
            l: np.load(os.path.join(cws_supp_dir, f"embedding_{l}.npy"))
            for l in ["en", "zh"]
        }
        pretrained_emb = pretrained_emb if lang is None else pretrained_emb[lang]
        token_to_ix = {
            l: json.load(open(os.path.join(cws_supp_dir, f"token2id_{l}.json"), "r"))
            for l in ["en", "zh"]
        }
        token_to_ix = token_to_ix if lang is None else token_to_ix[lang]

        text_processor = CWSProcessor(token_to_ix=token_to_ix)

    if debug:
        dataset = dataset[:100] if split else dataset["train"][:100]
    labels = []
    for d in dataset["train"]["answers"]:
        for a in d:
            labels.append(a["answer"])
    labels = (
        labels
        if lang is not None
        else [a["en"] for a in labels] + [a["zh"] for a in labels]
    )
    unique_labels = list(set(labels))

    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    processor = VTQAProcessor(
        text_question_merge=text_question_merge,
        text_processor=text_processor,
        img_processor=img_processor,
        img_type=img_feature_type,
        lang=lang,
    )

    def preprocess(examples):
        if img_feature_type == "image":
            images = examples["image_path"]
        elif img_feature_type == "region":
            images = [np.load(i)["x"].transpose((1, 0)) for i in examples["image_path"]]
        elif img_feature_type == "grid":
            images = [np.load(i).transpose((1, 0)) for i in examples["image_path"]]

        encoding = processor(
            contexts=examples["context"],
            questions=examples["question"],
            images=images,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
            max_context_token=max_context_token,
            max_question_token=max_question_token,
        )

        targets = []
        if cls_target:
            SCORE_PER_MATCH = 1

            for answers in examples["answers"]:
                target = np.zeros(len(id2label))
                if answers is not None:
                    for a in answers:
                        if a["answer"] in label2id:
                            target[label2id[a["answer"]]] = min(
                                1, target[label2id[a["answer"]]] + SCORE_PER_MATCH
                            )

                targets.append(target)

            encoding["labels"] = targets
        else:
            # TODO: label tokenize
            # LlamaTokenizer()(text='', text_target='')

            encoding["labels"] = text_processor(
                text=examples["answer"],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                max_length=max_answer_token,
            )["input_ids"]
        return encoding

    dataset = dataset.map(preprocess, batched=True, batch_size=40, num_proc=4)
    remove_columns = [
        "question",
        "context",
        "image_id",
        "image_path",
        "answers",
        "cws_path",
    ]
    torch_columns = (
        [
            "question_id",
            "input_ids",
            "attention_mask",
            "labels",
            "pixel_values",
            "image_attention_mask",
        ]
        if text_question_merge
        else [
            "question_id",
            "context_input_ids",
            "question_input_ids",
            "context_attention_mask",
            "question_attention_mask",
            "labels",
            "pixel_values",
            "image_attention_mask",
        ]
    )
    dataset = dataset.remove_columns(remove_columns)
    dataset.set_format(type="torch", columns=torch_columns)
    return dataset, token_to_ix, pretrained_emb, label2id, id2label


class VTQAProcessor:
    def __init__(
        self,
        text_question_merge=False,
        text_processor=None,
        img_processor=None,
        img_type="image",
        lang: str = "en",
    ) -> None:
        self.text_question_merge = text_question_merge
        self.img_type = img_type

        self.text_processor = text_processor

        self.img_processor = img_processor

    def __call__(
        self,
        contexts=None,
        questions=None,
        images=None,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=None,
        max_context_token=None,
        max_question_token=None,
        max_answer_token=None,
        *args,
        **kwds,
    ):

        if self.img_processor and self.img_type == "image":
            ip = self.img_processor(images)
            images = ip["pixel_values"]
            image_attention_mask = ip["pixel_mask"]
        else:
            assert self.img_type in ["region", "grid"], "wrong img_type"
            img_feat_pad_size = 100 if self.img_type == "region" else 608

            images, image_attention_mask = padding_np(
                images, img_feat_pad_size, 0, 2, mode="constant", constant_values=0
            )
        if self.text_question_merge:
            # TODO: llava prompt
            # texts = [ct + " " + q for ct, q in zip(contexts, questions)]
            texts = [
                "<image>\nUSER: " + ct + " " + q for ct, q in zip(contexts, questions)
            ]
            text_encoding = self.text_processor(
                text=texts,
                padding=padding,
                truncation=truncation,
                return_tensors=return_tensors,
                max_length=max_length,
            )

            if isinstance(
                self.text_processor, (PreTrainedTokenizer, PreTrainedTokenizerFast)
            ):
                for k, v in text_encoding.items():
                    text_encoding[k] = v.squeeze()

            return {
                **text_encoding,
                "pixel_values": images,
                "image_attention_mask": image_attention_mask,
            }
        else:
            context_encoding = self.text_processor(
                text=contexts,
                padding=padding,
                truncation=truncation,
                return_tensors=return_tensors,
                max_length=max_context_token,
            )
            question_encoding = self.text_processor(
                text=questions,
                padding=padding,
                truncation=truncation,
                return_tensors=return_tensors,
                max_length=max_question_token,
            )

            if isinstance(
                self.text_processor, (PreTrainedTokenizer, PreTrainedTokenizerFast)
            ):
                for encoding in [context_encoding, question_encoding]:
                    for k, v in encoding.items():
                        encoding[k] = v.squeeze()
            return {
                "context_input_ids": context_encoding["input_ids"],
                "context_attention_mask": context_encoding["attention_mask"],
                "question_input_ids": question_encoding["input_ids"],
                "question_attention_mask": question_encoding["attention_mask"],
                "pixel_values": images,
                "image_attention_mask": image_attention_mask,
            }


class CWSProcessor:
    def __init__(
        self,
        token_to_ix=None,
    ) -> None:
        self.token_to_ix = token_to_ix

    def __call__(
        self,
        text=None,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=None,
    ):
        if isinstance(text, (list, tuple)):
            context_ids = [
                [
                    (
                        self.token_to_ix[w]
                        if w in self.token_to_ix
                        else self.token_to_ix["UNK"]
                    )
                    for w in t
                ]
                for t in text
            ]
        else:
            context_ids = [self.token_to_ix(w) for w in text]
        if truncation:
            if padding == "max_length":
                assert max_length, "padding=max_length, but max_length is not specified"
                context_ids, context_ids_mask = padding_np(
                    context_ids,
                    max_length,
                    0,
                    1,
                    mode="constant",
                    constant_values=self.token_to_ix["PAD"],
                )

        return {
            "input_ids": context_ids,
            "attention_mask": context_ids_mask,
        }


def padding_np(feat, max_l, pad_dim, all_dim, mode="constant", constant_values=0):
    feat = [np.asarray(i[:max_l]) for i in feat]
    new_feat = []
    for f in feat:
        pad_with = [(0, 0)] * all_dim
        pad_with[pad_dim] = (0, max_l - f.shape[0])
        f = np.pad(
            f,
            pad_with,
            mode=mode,
            constant_values=constant_values,
        )
        new_feat.append(f)
    if all_dim == 2:
        feat_attention_mask = [np.sum(np.abs(f), axis=-1) == 0 for f in new_feat]
    elif all_dim == 1:
        feat_attention_mask = [f == 0 for f in new_feat]
    return new_feat, feat_attention_mask
