import torch
import numpy as np
import torch.nn as nn
import logging
import argparse
import time
import torch.nn.functional as F

from torch.autograd import Variable
from torchvision import datasets, transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def get_images(n_samples,r:int=8,c:int=8,batch_size:int=5):
    # Concentrating on the first 100 samples
    n_samples = n_samples
    rows=r
    cols=c
# training data
    X_train = datasets.MNIST(root='./data', train=True, download=True,
                             transform=transforms.Compose([transforms.Grayscale(),
                                                           transforms.ToTensor(),
                                                           # transforms.Normalize((0.1307,), (0.3081,)),
                                                           # transforms.Resize([rows, cols])
                                                           ]))

    # Leaving only labels 0 and 1
    idx = np.append(np.where(X_train.targets == 0)[0][:n_samples],
                    np.where(X_train.targets == 1)[0][:n_samples])

    X_train.data = X_train.data[idx]
    X_train.targets = X_train.targets[idx]

    train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size , shuffle=True)

# test data
    X_test = datasets.MNIST(root='./data', train=False, download=True,
                            transform=transforms.Compose([transforms.Grayscale(),
                                                          transforms.ToTensor(),
                                                          # transforms.Resize([rows, cols])
                                                          ]))

    idx = np.append(np.where(X_test.targets == 0)[0][:n_samples],
                    np.where(X_test.targets == 1)[0][:n_samples])

    X_test.data = X_test.data[idx]
    X_test.targets = X_test.targets[idx]

    test_loader = torch.utils.data.DataLoader(X_test, batch_size=n_samples*10, shuffle=True)
    return train_loader,test_loader,rows*cols


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 2 classes
        self.fc = nn.Linear(1568, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        output=F.log_softmax(x,dim=1)
        return output, x    # return x for visualization

def train(num_epochs,loaders,epoch):
    # loss_func = nn.()
    cnn.train()
    # Train the model
    total_step = len(loaders['train'])

    for i, (images, labels) in enumerate(loaders['train']):
        # gives batch data, normalize x when iterate train_loader
        b_x = Variable(images)  # batch x
        b_y = Variable(labels)  # batch y

        output = cnn(b_x)[0]
        # print(b_y.shape)
        # print(output.shape)

        loss = F.nll_loss(output,b_y)
        # clear gradients for this training step
        optimizer.zero_grad()
        # backpropagation, compute gradients
        loss.backward()
        # apply gradients
        optimizer.step()
        # print(i)
        if (i+1)%5==0:
            print('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item())
                  )

def test(loaders):
    cnn.eval()
    test_loss=0
    correct = 0
    with torch.no_grad():
        for i,(data, target) in enumerate(loaders['test']):
            output = cnn(data)[0]
            test_loss += F.nll_loss(output, target,reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    # test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--method', type=str, help='FRQI/angle/amplitude', default="CNN")
    parser.add_argument('--sample', type=int, help='number of samples', default=50)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.01)
    parser.add_argument('--batch', type=int, help='batch size', default=5)
    parser.add_argument('--epoch', type=int, help='epoch numbers', default=20)
    parser.add_argument('--layer', type=int, help='ansatz layer numbers', default=1)
    args = vars(parser.parse_args())
    # set hyper parameters
    embedding_methods = args['method']
    sample_number = args['sample']
    batch_size = args['batch']
    learning_rate = args['lr']
    epoch_number = args['epoch']
    layer_number = args['layer']

    print(embedding_methods, sample_number, batch_size, learning_rate, epoch_number, layer_number)

    # set file name
    time_stamp=time.strftime("%d-%H%M%S", time.localtime())
    file_name = f"{embedding_methods}_{time_stamp}"
    # set logging config
    logging.basicConfig(level=logging.INFO,
                        filename=f'./data/loggings/{file_name}.log')
    logging.info(f"embedding methods {embedding_methods}, "
                 f"sample number {sample_number}, "
                 f"batch size {batch_size}, "
                 f"learning rate {learning_rate}, "
                 f"epoch number {epoch_number}, "
                 f"layer number {layer_number}")

    train_loader, test_loader, image_size = get_images(n_samples=sample_number,
                                                       batch_size=batch_size)
    loaders = {
        'train': train_loader,
        'test': test_loader
    }

    # training part
    cnn = CNN()
    num_epochs = 10
    optimizer = torch.optim.NAdam(cnn.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        train(num_epochs, loaders,epoch)
        test(loaders)
