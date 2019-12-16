# Hardware accelerated Deep Neural Network
This repository hold's the sources of my bachelorthesis at Technical University 
(TU) Berlin written at DAI-Laboratory.
## Introduction
In this thesis the usabillity of an Intel Neural Compute Stick (NCS) for 
autonomous agents like NAO robot is evaluated. To get a qualified, meaningful 
result the runtime and accuracy of a Neural Net (NN) running on NCS aswell on 
CPU/GPU is measured. In the following sections the mainpart of written code is 
described.

### Digit recognition network
To measure the runtime a test-net is written in Python using PyTorch library. 
It is trained to identify handwritten digits with training- and dataset of 
MNIST. The NN is exported to ONNX format which is a necessary step to be 
understandable by Inference Engine (IE) of Intel OpenVino. OpenVino is a 
development toolkit which serves a python API for Intel NCS.

### Conversion of network model
The conversion of the NN is executed by a tool of OpenVino. 
As input the exported NN in ONXX format is used. 
As output a xml and bin file is generated which represents the weight and the 
structure of the NN.
To convert the model follow instruction [here](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_ONNX.html).

### Performance measurement of Intel NCS
The benchmarktest uses IE to run the exported NN. The time aswell the accuracy 
is the result of the benchmarktest. As CLI parameter the batchsize and the 
target device (CPU, GPU, NCS) is configurable.

## Setup the environment
This project is tested under python 3.7. You also need Intel OpenVino installed.
* python3.7 -m venv work3.7
* source ~/work3.7/bin/activate
* source ~/work3.7/bin/activate
* source /opt/intel/openvino/bin/setupvars.sh
