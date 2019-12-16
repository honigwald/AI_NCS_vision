import sys, os
import numpy as np
import logging as log
import matplotlib.pyplot as plt
import argparse
from time import time
from openvino.inference_engine import IENetwork, IECore, IEPlugin
from torchvision import datasets, transforms
from torch import nn, optim, utils

def data_loader(bs):
    # Loading MNIST Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ])
    valset = datasets.MNIST('./data/', download=True, train=False, transform=transform)
    valloader = utils.data.DataLoader(valset, batch_size=bs, shuffle=True)

    #images, labels = next(iter(valloader))
    return next(iter(valloader))

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    ### Parse CLI parameters
    parser = argparse.ArgumentParser(add_help=False, description='Benchmark sample NN on VPU_CPU')
    parser.add_argument('-d', '--device', 
                        help="Specify on which device the NN will be executed [CPU, MYRIAD]", 
                        required=True, type=str)
    parser.add_argument('-b', '--batchsize', 
                        help="Specify the batchsize (number of images to test)", 
                        required=True, type=int)
    parser.add_argument('-h', '--help',
                        action='help',
                        default=argparse.SUPPRESS,
                        help="Show this message.")

    args = parser.parse_args()
    device = args.device.upper()
    batch_size = args.batchsize

    ### Path to network model
    model_xml = "./model/digitrec_net.xml"
    model_bin = "./model/digitrec_net.bin"

    ### Plugin initialization for specified device and load extensions library if specified
    log.info("Creating Inference Engine")
    ie = IECore()

    ### Loading Network
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    ### loading testdata
    log.info("Loading testdata")
    images, labels = data_loader(batch_size)

    ### Prepare input blobs
    log.info("Preparing input blobs")
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))

    ### Loading model to the plugin
    log.info("Loading model to the plugin")
    plugin = IEPlugin(device=device)
    exec_net = plugin.load(network=net, num_requests=5)

    ### Processing

    ### Start sync inference
    log.info("Starting inference in synchronous mode")

    log.info("Processing input batch [Device: {}, Batchsize: {}]".format(device, batch_size))
    correct_proc = 0
    t_start = time()
    for i in range(batch_size):
        img = images[i].view(1, 784)
        res = exec_net.infer({input_blob: img})

        ### Processing output blob
        out = res[out_blob]
        prediction = np.exp(out[0])
        pred_digit = np.argmax(prediction)

        if labels[i].numpy() == pred_digit:
            correct_proc += 1
    t_end = time()

    log.info("Print the result")
    print("Total processed images: ", batch_size)
    print("Correct processed images: ", correct_proc)
    print("Accuracy: {:.2f}%".format((100/batch_size)*correct_proc))
    print("Time: {:.4f}s".format(t_end - t_start))

    imgplot = plt.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    plt.savefig('result/out.pdf', format='pdf')

    log.info("Finished sucessfully")

if __name__ == '__main__':
    sys.exit(main() or 0)
