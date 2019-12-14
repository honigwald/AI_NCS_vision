# Hardware accelerated Deep Neural Network
This repository hold's the sources of my Bachelorthesis at Technical University (TU) Berlin.
## Introduction
In this thesis the usabillity of an Intel Neural Compute Stick (NCS) for autonomous agents like NAO robot is evaluated. 
To get a qualified, meaningful result the runtime and accuracy of a Neural Net (NN) running on NCS aswell on CPU/GPU is measured.
In the following sections the mainpart of written code is described.

### Digit recognition network
To measure the runtime a test-net is written in Python using PyTorch library. 
It is trained to identify handwritten digits with training- and dataset of MNIST.
The NN is exported to ONXX format which is a necessary step to be understandable by Inference Engine (IE) of Intel OpenVino.
OpenVino is a development toolkit which serves a python API for Intel NCS.

### Conversion of network model
The conversion of the NN is executed by a tool of OpenVino. 
As input the exported NN in ONXX format is used. 
As output a xml and bin file is generated which represents the weight and the structure of the NN.

### Performance measurement of Intel NCS
The benchmarktest uses IE to run the exported NN. The time aswell the accuracy is the result of the benchmarktest. 
As CLI parameter the batchsize and the target device (CPU, GPU, NCS) is configurable.

## Setup runtime-environment
source work3.7/bin/activate
python3.7 -m venv work3.7
