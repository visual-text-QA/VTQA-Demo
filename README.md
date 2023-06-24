# VTQA-Demo

This repository is used as a demo for the [VTQA Challenge](https://visual-text-qa.github.io/)

For more details about the method, please refer to the [paper](https://arxiv.org/abs/2303.02635)

## Table of Contents

0. [Install Docker](#Prerequisites)
0. [Pull the image](#Training)
0. [Download dataset](#Validation-and-Testing)
0. [Create container](#Pretrained-models)
0. [Clone demo code](#Pretrained-models)
0. [Train, val & test](#citation)
0. [Submmit](#citation)

## Install Docker

Go to <https://docs.docker.com/get-docker/> and install the Docker application corresponding to your platform.

## Pull the image

```
docker pull nvcr.io/nvidia/pytorch:21.12-py3
```

(You can also use the other images to do you work.But our demo has only been tested in this image.)

## Download dataset

You can register on our challenge [website](http://vtqa-challenge.fixtankwun.top:20010/) and download the data.

Unzip the files and place them as follows:

```angular2html
|-- data
    |-- images
    |  |-- train
    |  |-- val
    |  |-- test_dev
    |-- annotations
```

## Create container

```
docker run --gpus all -itd --shm-size 8g --name vtqa -v /your-data-path/:/workspace/data nvcr.io/nvidia/pytorch:21.12-py3
```

```
docker exec -it vtqa /bin/bash
```

## Clone demo code

```
cd /workspace
git clone https://github.com/visual-text-QA/VTQA-Demo.git
```

## Train, val & test

```
cd VTQA-Demo
python main.py --RUN train
```

The default setting for `train` will eval the val set every epoch and eval the test_dev set after training. The trained model will be saved in `/workspace/vtqa/results/ckpts/ckpt_demo/epoch13.pkl`. And the predict answers for the test_dev set will be saved in `/workspace/vtqa/results/pred/test_dev_result_demo.json`.

Tips: In this demo, we use the word vectors that are segmented and vectorized by spaCy([zh_core_web_lg:3.3.0](https://github.com/explosion/spacy-models/releases/tag/zh_core_web_lg-3.3.0) and [en_core_web_lg:3.3.0](https://github.com/explosion/spacy-models/releases/tag/en_core_web_lg-3.3.0)). If necessary, you can conduct the word segmentation and vectorization yourself, and modify the corresponding code yourself.

## Submitting

Register and login [here](http://vtqa-challenge.fixtankwun.top:20010/).

For test_dev set, just download the `test_dev_result_demo.json` file and upload in the challenge website.

```
docker cp vtqa:/workspace/VTQA-Demo/results/pred/test_dev_result_demo.json /your-save-path/
```

For test set, you need to commit you container to a docker image and push your image to [DockerHub](https://hub.docker.com/). (Other public Docker image sources can also be used)

Example for a Dockerhub user called `<username>` and Docker Image name `<imagename>` (example for Image name: `vtqa:submission`):
```
SOURCE=<container name>
USER=<username>
IMAGENAME=<imagename>
docker login
docker commit -a ${USER} -m "vtqa submission" ${SOURCE} ${IMAGENAME}
docker push ${IMAGENAME}
```

Tips: To avoid code leakage, it is recommended to use random image names or private images

Then you can submit your docker image name [here](http://vtqa-challenge.fixtankwun.top:20010/).

Your submission will be run using the following command: 

```
docker pull <username>/<imagename>
docker run --network none --shm-size 8g -v <path to folder containing the test.json>:/workspace/data -v <path to save predict json>:/workspace/test_pred.json --gpus 0 --rm <username>/<imagename> /bin/bash /workspace/VTQA-Demo/test.sh
```

To ensure correct submission, it is recommended that you conduct local testing through the following commands before submitting

```
DATA=/your-data-path/
cp ${DATA}/annotations/test_dev_en.json ${DATA}/annotations/test_en.json 
cp ${DATA}/annotations/test_dev_zh.json ${DATA}/annotations/test_zh.json 
cp ${DATA}/annotations/test_dev_cws_en.json ${DATA}/annotations/test_cws_en.json 
cp ${DATA}/annotations/test_dev_cws_zh.json ${DATA}/annotations/test_cws_zh.json
touch ${DATA}/test_pred.json
docker run --network none --shm-size 8g -v ${DATA}:/workspace/data -v ${DATA}/test_pred.json:/workspace/test_pred.json --gpus 0 --rm <username>/<imagename> /bin/bash /workspace/VTQA-Demo/test.sh
```

## Citation

If this repository is helpful for your research, we'd really appreciate it if you could cite the following paper:

```
@inproceedings{Chen2023VTQAVT, 
    title={VTQA: Visual Text Question Answering via Entity Alignment and Cross-Media Reasoning}, 
    author={Kang Chen and Xiangqian Wu}, 
    year={2023} 
}
```
