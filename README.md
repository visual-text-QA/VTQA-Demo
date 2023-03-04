# VTQA-Demo

This repository is used as a demo for the [VTQA Challenge](https://visual-text-qa.github.io/)

For more details about the method, please refer to the [paper](https://arxiv.org/abs/***)

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
docker pull nvcr.io/nvidia/pytorch:23.02-py3
```

(You can also use the other images to do you work.But our demo has only been tested in this image.)

## Download dataset

Download the dataset from [GoogleDrive](https://drive.google.com/file/d/1UA2yYFvoWTjBoFrqKy6KxeIEnynd6qlP/view?usp=sharing) or [BaiduYun](https://pan.baidu.com/s/1mIHGO18Jhjyb2XHHsIGBeA?pwd=4dce).

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
docker run --gpus all -itd --name vtqa -v /your-data-path/:/workspace/data nvcr.io/nvidia/pytorch:23.02-py3
```

```
docker exec -it vtqa /bin/bash
```

You can now train your own model in this container. (Or just use the [demo](https://github.com/visual-text-QA/VTQA-Demo) for a quick start.)

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

## Submitting

Register and login [here](http://81.70.95.220:20035/).

For test_dev set, just upload the `test_dev_result_demo.json` file.

For test set, you need to commit you container to a docker image and push your image to [DockerHub](https://hub.docker.com/).

Example for a Dockerhub user called `<username>`:
```
SOURCE=<container name>
USER=<username>
docker login
docker commit -a $USER -m "vtqa submission" ${SOURCE}  vtqa:submission
docker push vtqa:submission
```

Then you can submit your docker image name [here](http://81.70.95.220:20035/).

Your submission will be run using the following command: 

```
docker pull <username>/vtqa:submission
docker run -v <path to folder containing the test.json>:/workspace/data -v <path to save predict json>:/workspace/test_pred.json --gpus all --rm vtqa:submission /bin/bash -c "python /workspace/VTQA-Demo/test.py"
```

## Citation

If this repository is helpful for your research, we'd really appreciate it if you could cite the following paper:

```

```
