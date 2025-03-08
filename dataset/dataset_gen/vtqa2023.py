# coding=utf-8
# Copyright 2022 The PolyAI and HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import numpy as np
import datasets

logger = datasets.logging.get_logger(__name__)

datasets.Image()
""" VTQA Dataset"""

_CITATION = """\
@inproceedings{chen2024vtqa,
  title={VTQA: Visual Text Question Answering via Entity Alignment and Cross-Media Reasoning},
  author={Chen, Kang and Wu, Xiangqian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={27218--27227},
  year={2024}
}
"""

_DESCRIPTION = """\
VTQA is a new dataset containing open-ended questions about image-text pairs. 
These questions require multimedia entity alignment, multi-step reasoning and open-ended answer generation.
"""

_HOMEPAGE_URL = "https://visual-text-qa.github.io/"

_LICENSE = "Creative Commons Attribution NonCommercial NoDerivs 4.0 International License"

# 修改数据URL，使用raw格式
_DATA_URL = "https://huggingface.co/datasets/CalfKing/vtqa2023/resolve/main/data"
# 或者使用这种格式
# _DATA_URL = "https://huggingface.co/datasets/CalfKing/vtqa2023/raw/main/data"

_ALL_CONFIGS = sorted(
    [
        "zh-image",
        "zh-region",
        "zh-grid",
        "en-image",
        "en-region",
        "en-grid",
        "en",
        "zh",
        "image",
        "region",
        "grid",
    ]
)

_BASE_IMAGE_FEATURES = {
    "image": datasets.Image(),
    "region": datasets.Value("string"),
    "grid": datasets.Value("string"),
}

_BASE_TEXT_FEATURES = {
    "raw": {
        "en": datasets.Value("string"),
        "zh": datasets.Value("string"),
    },
    "cws": {
        "en": [datasets.Value("string")],
        "zh": [datasets.Value("string")],
    },
}

_BASE_ANSWER_FEATURES = {
    "answer_type": datasets.Value("string"),
    "answer": {
        "en": datasets.Value("string"),
        "zh": datasets.Value("string"),
    },
}


class VTQAConfig(datasets.BuilderConfig):
    """BuilderConfig for VTQA."""

    def __init__(
        self, data_url: str = None, use_cws=False, local_url=None, get_test_split=False, **kwargs
    ):
        super(VTQAConfig, self).__init__(
            version=datasets.Version("1.0.0", ""),
            description=self.description,
            **kwargs,
        )
        self.data_url = _DATA_URL if data_url is None else data_url
        self.use_cws = use_cws
        self.local_url = local_url
        self.get_test_split = get_test_split
        self.cws_supp_dir = None

    @property
    def features(self):
        # 设置默认值
        lang, image_type = "all", "all"
        
        if self.name == "all":
            lang, image_type = "all", "all"
        elif "-" in self.name:
            lang, image_type = self.name.split("-")
        elif self.name in ["en", "zh"]:
            lang, image_type = self.name, "all"
        elif self.name in ["image", "region", "grid"]:
            lang, image_type = "all", self.name
        
        self.lang, self.image_type = lang, image_type

        btf = _BASE_TEXT_FEATURES["cws"] if self.use_cws else _BASE_TEXT_FEATURES["raw"]
        baf = {
            "answer_type": _BASE_ANSWER_FEATURES["answer_type"],
            "answer": (
                _BASE_ANSWER_FEATURES["answer"]
                if lang == "all"
                else _BASE_ANSWER_FEATURES["answer"][lang]
            ),
        }
        dataset_features = datasets.Features(
            {
                "question": (btf if lang == "all" else btf[lang]),
                "question_id": datasets.Value("int64"),
                "context": (btf if lang == "all" else btf[lang]),
                "image_id": datasets.Value("int64"),
                "image_path": (
                    _BASE_IMAGE_FEATURES
                    if image_type == "all"
                    else _BASE_IMAGE_FEATURES[image_type]
                ),
                "answers": [baf],
                "cws_path": datasets.Value("string"),
            }
        )
        return dataset_features


def _build_config(name, use_cws=False, local_url=None):
    return VTQAConfig(
        name=name,
        data_url=_DATA_URL,
        use_cws=use_cws,
        local_url=local_url,
    )


class VTQA(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIG_CLASS = VTQAConfig
    DEFAULT_WRITER_BATCH_SIZE = 1000
    BUILDER_CONFIGS = [_build_config(name) for name in _ALL_CONFIGS + ["all"]]

    def _info(self):

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=self.config.features,
            homepage=_HOMEPAGE_URL,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        lang, image_type = self.config.lang, self.config.image_type

        def _get_url(file_name):
            if self.config.local_url is not None:
                # 检查本地路径是否存在
                local_path = os.path.join(self.config.local_url, f"{file_name}")
                if os.path.exists(local_path):
                    return local_path
                else:
                    logger.warning(f"Local path {local_path} not found, falling back to download")
            
            # 如果local_url未指定或本地文件不存在,则从远程下载
            remote_url = os.path.join(self.config.data_url, f"{file_name}.zip")
            try:
                return dl_manager.download_and_extract(remote_url)
            except Exception as e:
                raise ValueError(f"Failed to download or extract {remote_url}: {str(e)}")

        annotation_dir = _get_url("annotations")
        image_dir, region_dir, grid_dir = None, None, None
        if image_type in ["image", "all"]:
            image_dir = _get_url("image")
        if image_type in ["region", "all"]:
            region_dir = _get_url("region")
        if image_type in ["grid", "all"]:
            grid_dir = _get_url("grid")

        if self.config.use_cws:
            cws_supp_dir = _get_url("cws_supp")
            self.config.cws_supp_dir = cws_supp_dir

        datasets_split = [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(annotation_dir, "train.json"),
                    "image_dir": (
                        os.path.join(image_dir, "train") if image_dir else None
                    ),
                    "region_dir": (
                        os.path.join(region_dir, "train") if region_dir else None
                    ),
                    "grid_dir": os.path.join(grid_dir, "train") if grid_dir else None,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(annotation_dir, "val.json"),
                    "image_dir": (
                        os.path.join(image_dir, "val") if image_dir else None
                    ),
                    "region_dir": (
                        os.path.join(region_dir, "val") if region_dir else None
                    ),
                    "grid_dir": os.path.join(grid_dir, "val") if grid_dir else None,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split("test_dev"),
                gen_kwargs={
                    "filepath": os.path.join(annotation_dir, "test_dev.json"),
                    "image_dir": (
                        os.path.join(image_dir, "test_dev") if image_dir else None
                    ),
                    "region_dir": (
                        os.path.join(region_dir, "test_dev") if region_dir else None
                    ),
                    "grid_dir": (
                        os.path.join(grid_dir, "test_dev") if grid_dir else None
                    ),
                    "labeled": False,
                },
            ),
        ]

        if self.config.get_test_split:
            return datasets_split + [
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "filepath": os.path.join(annotation_dir, "test.json"),
                        "image_dir": (
                            os.path.join(image_dir, "test") if image_dir else None
                        ),
                        "region_dir": (
                            os.path.join(region_dir, "test") if region_dir else None
                        ),
                        "grid_dir": (
                            os.path.join(grid_dir, "test") if grid_dir else None
                        ),
                        "labeled": False,
                    },
                )
            ]
        else:
            return datasets_split

    def _generate_examples(
        self, filepath, image_dir=None, region_dir=None, grid_dir=None, labeled=True
    ):
        # 添加文件存在性检查
        if not os.path.exists(filepath):
            raise ValueError(f"Annotation file not found: {filepath}")
        
        if image_dir and not os.path.exists(image_dir):
            raise ValueError(f"Image directory not found: {image_dir}")
        
        if region_dir and not os.path.exists(region_dir):
            raise ValueError(f"Region directory not found: {region_dir}")
        
        if grid_dir and not os.path.exists(grid_dir):
            raise ValueError(f"Grid directory not found: {grid_dir}")

        lang, image_type = self.config.lang, self.config.image_type
        use_cws = "cws" if self.config.use_cws else "raw"
        """Yields examples as (key, example) tuples."""
        with open(filepath, encoding="utf-8") as f:
            vtqa = json.load(f)
            for id_, d in enumerate(vtqa):
                text_dict = {
                    "question": (
                        d["question"][use_cws]
                        if lang == "all"
                        else d["question"][use_cws][lang]
                    ),
                    "context": (
                        d["context"][use_cws]
                        if lang == "all"
                        else d["context"][use_cws][lang]
                    ),
                }
                image_dict = {}
                if image_dir is not None:
                    image_dict["image"] = os.path.join(
                        image_dir, d["image_name"]["image"]
                    )
                if region_dir is not None:
                    image_dict["region"] = os.path.join(
                        region_dir, d["image_name"]["region"]
                    )
                if grid_dir is not None:
                    image_dict["grid"] = os.path.join(grid_dir, d["image_name"]["grid"])

                if labeled:

                    yield id_, {
                        "question_id": d["question_id"],
                        "image_id": d["image_id"],
                        "answers": [
                            {
                                "answer_type": a["answer_type"],
                                "answer": (
                                    a["answer"] if lang == "all" else a["answer"][lang]
                                ),
                            }
                            for a in d["answers"]
                        ],
                        **text_dict,
                        "image_path": (
                            image_dict
                            if image_type == "all"
                            else image_dict[image_type]
                        ),
                        "cws_path": self.config.cws_supp_dir,
                    }
                else:
                    yield id_, {
                        "question_id": d["question_id"],
                        "image_id": d["image_id"],
                        "answers": None,
                        **text_dict,
                        "image_path": (
                            image_dict
                            if image_type == "all"
                            else image_dict[image_type]
                        ),
                        "cws_path": self.config.cws_supp_dir,
                    }
