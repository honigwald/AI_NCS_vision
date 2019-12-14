from openvino.inference_engine import IENetwork, IEPlugin
import sys
import numpy as np
import torch, torchvision
from torchvision import datasets, transforms
from torch import nn, optim

#net = IENetwork(model="digreco.xml", weights="digreco.bin")
#print(net.batch_size)
#print(net.inputs['data'].shape)

# Loading MNIST Data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ])
valset = datasets.MNIST('data/', download=True, train=False, transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

images, labels = next(iter(valloader))
 
img = images[0].view(1, 784)
#with torch.no_grad():
#    logps = model(img)

def main():
    #######################  Device  Initialization  ########################
    #  Plugin initialization for specified device and load extensions library if specified
    plugin = IEPlugin(device="MYRIAD") 
    #########################################################################
    
    #########################  Load Neural Network  #########################
    #  Read in Graph file (IR)
    net = IENetwork.from_ir(model="digreco.xml", weights="digreco.bin")
    
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    #  Load network to the plugin
    exec_net = plugin.load(network=net)
    del net
    ########################################################################
    
    #########################  Obtain Input Tensor  ########################
    #  Obtain and preprocess input tensor (image)
    #  Read and pre-process input image  maybe we don't need to show these details
    #image = cv2.imread("input_image.jpg")
    #
    ##  Preprocessing is neural network dependent maybe we don't show this
    n, c, h, w = net.inputs[input_blob]
    #image = cv2.resize(image, (w, h))
    #image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    #image = image.reshape((n, c, h, w))
    ########################################################################
    
    ##########################  Start  Inference  ##########################
    #  Start synchronous inference and get inference result
    req_handle = exec_net.start_async(inputs={input_blob: img})
    ########################################################################
    
    ######################## Get Inference Result  #########################
    status = req_handle.wait()
    res = req_handle.outputs[out_blob]
    
    
    # Do something with the results... (like print top 5)
    top_ind = np.argsort(res[out_blob], axis=1)[0, -5:][::-1]
    for i in top_ind:
        print("%f #%d" % (res[out_blob][0, i], i))
    
    ###############################  Clean  Up  ############################
    del exec_net
    del plugin
    ########################################################################
  
  
if __name__ == '__main__':
  sys.exit(main() or 0)


print("Hello World")
