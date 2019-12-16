import sys, os
import numpy as np
import logging as log
import matplotlib.pyplot as plt
import torch
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

def main():
    log.basicConfig(format="[%(levelname)s] %(message)s", level=log.INFO, stream=sys.stdout)
    log.info("Creating digit recognition net")

    ### Get test- and trainigdata
    log.info("Loading test- and trainingdata")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ])
    trainset = datasets.MNIST('data/', download=True, train=True, transform=transform)
    valset = datasets.MNIST('data/', download=True, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

    dataiter = iter(trainloader)
    images, labels  = dataiter.next()

    ### Define NN parameter
    input_size = 784            # Inputimage: 28px x 28px (=784)
    hidden_sizes = [128, 64]
    output_size = 10            # Possible result (0 to 9)

    ### Define NN model
    log.info("Creating netmodel")
    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),          # x > 0 = x, x < 0 = 0
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_size),
            nn.Softmax(dim=1))  # LogSoftmax(x_i) = log(\frac{exp(x_i)}{\sum_j{exp(x_j)}})
    print(model)

    ### Prepare NN
    criterion = nn.NLLLoss()
    images, labels = next(iter(trainloader))
    images = images.view(images.shape[0], -1)

    logps = model(images)
    loss = criterion(logps, labels)

    log.info("Initializing netmodel weights")
    loss.backward()

    ### Train NN using gradient descent
    log.info("Starting training")
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    t_start = time()
    epochs = 15
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            images = images.view(images.shape[0], -1)   # Flatten image into vector
            optimizer.zero_grad()       # Training pass
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()             # Train netmodel by backprop
            optimizer.step()            # Optimize weights
            running_loss += loss.item()
        else:
            print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))

    t_end = time()
    log.info("Training completed in {:.2f} Minutes".format(((t_end)-(t_start))/60))

    ### Test the network
    images, labels = next(iter(valloader))  # get random image

    img = images[0].view(1, 784)
    img_lab = labels[0].numpy()
    with torch.no_grad():
        logps = model(img)
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    log.info("Testing netmodel.")
    log.info("Predicted digit: {}".format(probab.index(max(probab))))
    log.info("Processed digit: {}".format(img_lab))
    
    ### Export trained network
    filenamepath = "model/digreco_net.onxx"
    log.info("Saving trained netmodel to '{}'".format(filenamepath))
    torch.save(model.state_dict(), filenamepath)

if __name__ == '__main__':
    sys.exit(main() or 0)
