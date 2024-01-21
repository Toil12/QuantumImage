import torch
import numpy as np
import torch.nn as nn
import logging
import argparse
import time
import torch.nn.functional as F

from torch.autograd import Variable
from torchvision import datasets, transforms
from npy_append_array import NpyAppendArray
from torchviz import  make_dot

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
                                                           transforms.Resize([rows, cols])
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
                                                          transforms.Resize([rows, cols])
                                                          ]))
    test_data_factor=10
    idx = np.append(np.where(X_test.targets == 0)[0][:n_samples*test_data_factor],
                    np.where(X_test.targets == 1)[0][:n_samples*test_data_factor])
    # print(idx.shape)
    X_test.data = X_test.data[idx]
    X_test.targets = X_test.targets[idx]

    test_loader = torch.utils.data.DataLoader(X_test, batch_size=n_samples*test_data_factor, shuffle=True)
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
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 2 classes
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        output=F.log_softmax(x,dim=1)
        return output,x    # return x for visualization

def train(num_epochs,loaders,epoch):

    # loss_func = nn.()
    cnn.train()
    training_loss=0
    # Train the model
    total_step = len(loaders['train'])
    samples=0
    correct_epoch=0
    for i, (images, labels) in enumerate(loaders['train']):
        correct = 0
        # gives batch data, normalize x when iterate train_loader
        b_x = Variable(images)  # batch x
        b_y = Variable(labels)  # batch y
        # make_dot(cnn(b_x), params=dict(cnn.named_parameters())).render("cnn_torchviz", format="png")
        output = cnn(b_x)[0]
        # print(b_y)
        judege=np.zeros(shape=(output.shape[0],))
        for i in range(output.shape[0]):
            if output[i,0]>=output[i,1]:
                judege[i]=0
            else:
                judege[i]=1
        judege=torch.tensor(judege)
        training_loss += F.mse_loss(judege, b_y, reduction='sum').item()
        pred = output.data.max(1, keepdim=True)[1]
        samples+=len(b_y)
        correct += pred.eq(b_y.data.view_as(pred)).sum()
        correct_epoch+= pred.eq(b_y.data.view_as(pred)).sum()
        accuracy_train_iteration=correct / b_y.shape[0]
        loss = F.nll_loss(output,b_y)
        # clear gradients for this training step
        optimizer.zero_grad()
        # backpropagation, compute gradients
        loss.backward()
        # apply gradients
        optimizer.step()
        # print(i)
        if (i+1)%5==0:
            print('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'
                  .format(epoch + 1,
                          num_epochs, i + 1,
                          total_step, loss.item(),
                          accuracy_train_iteration)
                  )
            logging.info(f"Iter: {i+1}| Accuracy: {accuracy_train_iteration} | Loss: {loss}")
    training_loss/=samples
    return correct_epoch / samples,training_loss
def test(loaders):
    cnn.eval()
    test_loss=0
    correct = 0
    with torch.no_grad():
        for i,(data, target) in enumerate(loaders['test']):
            output = cnn(data)[0]
            #
            judege = np.zeros(shape=(output.shape[0],))
            for i in range(output.shape[0]):
                if output[i, 0] >= output[i, 1]:
                    judege[i] = 0
                else:
                    judege[i] = 1
            judege = torch.tensor(judege)
            #
            test_loss += F.mse_loss(judege, target,reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            # print(pred.shape)
            correct += pred.eq(target.data.view_as(pred)).sum()
    accuracy_test=correct / len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)

    return accuracy_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--method', type=str, help='FRQI/angle/amplitude', default="CNN")
    parser.add_argument('--sample', type=int, help='number of samples', default=50)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.01)
    parser.add_argument('--batch', type=int, help='batch size', default=5)
    parser.add_argument('--epoch', type=int, help='epoch numbers', default=50)
    parser.add_argument('--layer', type=int, help='ansatz layer numbers', default=1)
    parser.add_argument('--seed', type=float, help='numpy random seed', default=0)
    args = vars(parser.parse_args())
    # set hyper parameters
    embedding_methods = args['method']
    sample_number = args['sample']
    batch_size = args['batch']
    learning_rate = args['lr']
    epoch_number = args['epoch']
    layer_number = args['layer']
    seed=args['seed']
    num_epochs = args['epoch']

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
                 f"layer number {layer_number}, "
                 f"numpy random seed {seed}")

    train_loader, test_loader, image_size = get_images(n_samples=sample_number,
                                                       batch_size=batch_size)
    loaders = {
        'train': train_loader,
        'test': test_loader
    }

    # training part
    cnn = CNN()
    optimizer = torch.optim.NAdam(cnn.parameters(), lr=0.01)


    # start training
    npaa_record = NpyAppendArray(f"./data/loss_epoch/{file_name}_loss.npy")
    models_path=f"./data/model/{file_name}_models.npy"
    # 'epoch number' for each epoch model, last is the optimizer
    models={}
    for epoch in range(num_epochs):
        time_start=time.time()
        print("start epoch {}".format(epoch + 1))
        logging.info(f"start epoch {epoch + 1}")
        epoch_acc,epoch_cost=train(num_epochs, loaders,epoch)
        epoch_acc_val=test(loaders)
        time_end=time.time()
        time_cost=time_end-time_start
        record_item = np.array([[epoch_acc, epoch_cost, epoch_acc_val]])
        npaa_record.append(record_item)
        print("epoch {} : "
              "Accuracy {} , "
              "Loss {}, "
              "Time {}, "
              "Val Accuracy {}\n".format(epoch + 1, epoch_acc, epoch_cost, time_cost, epoch_acc_val))
        logging.info(f"epoch {epoch + 1} : "
                     f"Training Accuracy {epoch_acc} , "
                     f"Loss {epoch_cost}, "
                     f"Time {time_cost} minutes, "
                     f"Validation Accuracy {epoch_acc_val}")

        # put the model into dictionary
        model=cnn.state_dict()
        models[epoch]=model
    models['optimizer']=optimizer.state_dict()
    npaa_record.close()
    torch.save(models,models_path)
