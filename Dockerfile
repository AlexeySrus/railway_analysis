FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# configure timezone, our app depends on it.
RUN /usr/bin/ln -sf /usr/share/zoneinfo/America/Toronto /etc/localtime

# install software
RUN apt update \
    && apt -y install python3-dev python3-pip ffmpeg unzip wget curl cmake screen git libturbojpeg \
    && apt clean

RUN pip3 install -U pip
RUN pip3 install --no-cache-dir Cython python-multipart

WORKDIR /app
COPY . .

RUN pip3 install onnx onnxruntime==1.16.0

RUN pip3 install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/app

EXPOSE 8052