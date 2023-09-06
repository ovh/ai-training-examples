# import dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms


# define neural network
class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


# define the model training function
def train_model(model, device, train_loader, optimizer, epoch):

    # launch model training
    model.train()

    # set up variables
    running_loss=0
    correct=0
    total=0

    # train the model on training data
    for data in train_loader:

        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.nll_loss(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    # calculate loss and accuracy on training set
    train_loss = running_loss/len(train_loader)
    accu = 100.*correct/total

    # add loss and accuracy into dedicated list
    train_losses.append(train_loss)
    train_accu.append(accu)

    # print result for each epoch
    print('Training - Loss: %.3f | Accuracy: %.3f'%(train_loss, accu))

    return


# define the model evaluation
def test_model(model, device, test_loader):

    # launch model DenseNet evaluation
    model.eval()

    running_loss=0
    correct=0
    total=0

    # evaluate the model on validation data
    with torch.no_grad():

        for data in test_loader:

            images,labels=data[0].to(device), data[1].to(device)

            outputs = model(images)
            loss = F.nll_loss(outputs, labels)
            running_loss+=loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    # calculate loss and accuracy on validation set
    test_loss = running_loss/len(test_loader)
    accu = 100.*correct/total

    # add loss and accuracy into dedicated list
    test_losses.append(test_loss)
    test_accu.append(accu)

    # print result for each epoch
    print('Validation - Loss: %.3f | Accuracy: %.3f'%(test_loss,accu))

    return


# load MNIST dataset and define data loaders
def load_data():

    # train loader
    train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/workspace/data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
        batch_size=64, shuffle=True)

    # test loader
    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/workspace/data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=64, shuffle=True)

    return train_loader, test_loader


# check if cuda is available
def check_gpu():

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('Running on GPU...')

    else:
        device = torch.device('cpu')
        print('Running on CPU...')

    return device


# export model to ONNX
def export_onnx(model):

    # load pytorch model
    model.load_state_dict(torch.load('/workspace/models/model_mnist_classification.pt'))

    # pt model to onnx
    dummy_input = torch.randn(1, 1, 28, 28, device="cuda")
    torch.onnx.export(model, dummy_input, "/workspace/models/model_mnist_classification.onnx")

    return


# define main
if __name__ == '__main__':

    # load data
    train_loader, test_loader = load_data()

    # define device
    device =  check_gpu()

    # initialize model
    model = Network().to(device)

    # set optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    # define the empty lists to add losses and accuracies
    train_losses, train_accu, test_losses, test_accu = ([] for i in range(4))

    # launch model training and evaluation
    for epoch in range(0, 10):

        print('\nEpoch %d/%d:'%(epoch, 10))

        train_model(model, device, train_loader, optimizer, epoch)
        test_model(model, device, test_loader)

    # save pytorch model
    torch.save(model.state_dict(),"/workspace/models/model_mnist_classification.pt")

    # export the model to onnx
    export_onnx(model)
