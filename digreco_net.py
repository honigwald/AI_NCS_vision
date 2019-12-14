import numpy as np
import torch
import torchvision
import logging as log
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

### Plot given image
def view_classify(img, ps):
    ps = ps.data.numpy().squeeze()
    fi, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probalitiy')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()

def main():
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
    log.info("Creating network model")
    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),          # x > 0 = x, x < 0 = 0
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_size),
            nn.Softmax(dim=1))  # LogSoftmax(x_i) = log(\frac{exp(x_i)}{\sum_j{exp(x_j)}})

    log.info(model)

    ### Prepare NN
    criterion = nn.NLLLoss()
    images, labels = next(iter(trainloader))
    images = images.view(images.shape[0], -1)

    logps = model(images)
    loss = criterion(logps, labels)

    log.info("Initializing network weights")
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
        log.info("Training completed")
        print("\nTraining Time: {} Minutes".format(t_end-t_start)/60)

    images, labels = next(iter(valloader))

    img = images[0].view(1, 784)
    with torch.no_grad():
        logps = model(img)

    ### Test the network
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    print("Predicetd Digit: ", probab.index(max(probab)))
    view_classify(img.view(1, 28, 28), ps)

    ### Export trained network
    log.info("Saving trained network model")
    torch.save(model.state_dict(), 'model/digreco_net.onxx')

if __name__ == '__main__':
    sys.exit(main() or 0)
