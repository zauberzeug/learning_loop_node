#!/bin/bash

# Taken from: https://github.com/daisukekobayashi/darknet-docker/blob/master/configure.sh
tag="${1}"
makefile="Makefile"

function enable_gpu() {
  echo "enable GPU flag"
  sed -i -e 's/GPU=0/GPU=1/g' $makefile
}

function enable_cudnn() {
  echo "enable CUDNN flag"
  sed -i -e 's/CUDNN=0/CUDNN=1/g' $makefile
}

function enable_cudnn_half() {
  echo "enable CUDNN_HALF flag"
  sed -i -e 's/CUDNN_HALF=0/CUDNN_HALF=1/g' $makefile
}

function enable_opencv() {
  echo "enable OPENCV flag"
  sed -i -e 's/OPENCV=0/OPENCV=1/g' $makefile
}

function enable_libso() {
  echo "enable LIBSO flag"
  sed -i -e 's/LIBSO=0/LIBSO=1/g' $makefile
}

function enable_debug() {
  echo "enable DEBUG flag"
  sed -i -e 's/DEBUG=0/DEBUG=1/g' $makefile
}

function enable_arch() {
  echo "enable ARCH= -gencode arch=compute_$1,code[sm_$1,compute_$1"
  sed -i -e "s/# ARCH= -gencode arch=compute_$1/ARCH= -gencode arch=compute_$1/g" $makefile
}

case ${tag} in
    "cpu")
    ;;
    "gpu")
    enable_gpu
    ;;
    "cudnn")
    enable_gpu
    enable_cudnn
    ;;
    "cudnn-half")
    enable_gpu
    enable_cudnn
    enable_cudnn_half
    ;;
    "gpu-cc75")
    enable_gpu
    enable_cudnn
    enable_cudnn_half
    enable_arch 75
    ;;
    "gpu-cv-cc75")
    enable_gpu
    enable_cudnn
    enable_cudnn_half
    enable_opencv
    enable_arch 75
    ;;
    *)
    echo "error: $tag is not supported"
    exit 1
    ;;
esac
