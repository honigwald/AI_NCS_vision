import sys, os
import numpy as np
import torch
import torchvision
import logging as log
import matplotlib.pyplot as plt
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
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)

            # Training pass
            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)

            # This is where the model learns by backpropagation
            loss.backward()

            # And optimizes its weights here
            optimizer.step()

            running_loss += loss.item()
        else:
            print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))

    t_end = time()
    log.info("Training completed in {:.2f} Minutes".format(((t_end)-(t_start))/60))

    ### Test the network
    # get random image
    images, labels = next(iter(valloader))

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
