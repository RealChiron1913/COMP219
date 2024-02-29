import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import argparse
import time
import copy


# define fully connected network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        output = F.log_softmax(x, dim=1)
        return output


################################################################################################
## Note: below is the place you need to add your own attack algorithm
################################################################################################
def random_attack(model, X, y, device):
    X_adv = Variable(X.data)

    random_noise = torch.FloatTensor(*X_adv.shape).uniform_(-0.1, 0.1).to(device)
    X_adv = Variable(X_adv.data + random_noise)

    return X_adv


def eval_adv_test(model, device, test_loader, adv_attack_method):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), 28 * 28)
            adv_data = adv_attack_method(model, data, target, device=device)
            output = model(adv_data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def evaluate_all_models(model_file, attack_method, test_loader, device):
    model = Net().to(device)
    model.load_state_dict(torch.load(model_file))

    adv_attack = attack_method
    ls, acc = eval_adv_test(model, device, test_loader, adv_attack)

    del model
    return 1 / acc, acc


def adv_attack(model, X, y, device):
    X_adv = Variable(X.data)

    ################################################################################################
    ## Note: below is the place you need to edit to implement your own attack algorithm
    ################################################################################################

    # random_noise = torch.FloatTensor(*X_adv.shape).uniform_(-0.1, 0.1).to(device)
    # X_adv = Variable(X_adv.data + random_noise)

    epsilon = 0.108
    step_size = 0.006
    num_steps = 100

    random_noise = torch.FloatTensor(*X_adv.shape).uniform_(-epsilon, epsilon).to(device)
    processed_data = Variable(X_adv.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([processed_data], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(processed_data), y)
        loss.backward()
        eta = step_size * processed_data.grad.data.sign()
        processed_data = Variable(processed_data.data + eta, requires_grad=True)
        eta = torch.clamp(processed_data.data - X.data, -epsilon, epsilon)
        processed_data = Variable(X.data + eta, requires_grad=True)
        X_adv = Variable(torch.clamp(processed_data, 0, 1.0), requires_grad=True)
    ################################################################################################
    ## end of attack method
    ################################################################################################

    return X_adv


def main():
    ################################################################################################
    ## Note: below is the place you need to load your own attack algorithm and defence model
    ################################################################################################

    # attack algorithms name, add your attack function name at the end of the list
    attack_method = [random_attack, adv_attack]

    # defense model name, add your attack function name at the end of the list
    model_file = ["1000.pt", "201536436.pt"]

    ################################################################################################
    ## end of load
    ################################################################################################

    # number of attack algorithms
    num_all_attack = len(attack_method)
    # number of defense model
    num_all_model = len(model_file)

    # define the evaluation matrix number
    evaluation_matrix = np.zeros((num_all_attack, num_all_model))

    # setup training parameters
    parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    args = parser.parse_args(args=[])

    # judge cuda is available or not
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")
    device = torch.device("cpu")

    torch.manual_seed(args.seed)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_set = torchvision.datasets.FashionMNIST(root='data', train=True, download=True,
                                                  transform=transforms.Compose([transforms.ToTensor()]))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    test_set = torchvision.datasets.FashionMNIST(root='data', train=False, download=True,
                                                 transform=transforms.Compose([transforms.ToTensor()]))
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    for i in range(num_all_attack):
        for j in range(num_all_model):
            attack_score, defence_score = evaluate_all_models(model_file[j], attack_method[i], test_loader, device)
            evaluation_matrix[i, j] = attack_score

    print("evaluation_matrix: ", evaluation_matrix)
    # Higher is better
    print("attack_score_mean: ", evaluation_matrix.mean(axis=1))
    # Higher is better
    print("defence_score_mean: ", 1 / evaluation_matrix.mean(axis=0))


main()
