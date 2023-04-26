ARG BASE_IMG=pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
FROM $BASE_IMG

WORKDIR /workspace/proj

ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

COPY ./requirements.txt /workspace

RUN pip install --upgrade pip wheel setuptools
RUN apt-get update
RUN apt-get -y install gcc g++
RUN pip install -r /workspace/requirements.txt

ADD ./SPIGA /workspace/SPIGA
RUN pip install -e /workspace/SPIGA

RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install torch==1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
