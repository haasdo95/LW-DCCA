"""
This file contains all that is needed to run dcca on the left and right half of mnist numbers
"""

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
import argparse
from tensorboardX import SummaryWriter
import numpy as np

from dcca_loss import CorrLoss

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--left-width', type=int, default=2038, metavar='N',
                    help='width of linear layer on the left (default: 2038)')
parser.add_argument('--right-width', type=int, default=1608, metavar='N',
                    help='width of linear layer on the right (default: 1608)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--use-cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--gpu_num', type=int, default=0, metavar='S',
                    help='decides which gpu to use')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20,
                    help='how many batches to wait before logging training status')
parser.add_argument('--log-dir', type=str, metavar='N',
                    help='directory to store logs')
parser.add_argument('--weight-interval', type=int, default=100,
                    help='how many batches to wait before saving model weights')
parser.add_argument('--weight-dir', type=str, metavar='N',
                    help='directory to store model weights')
parser.add_argument('--cca-reg', type=float,
                    help='<=0 if using Ledoit')
parser.add_argument('--mu-gradient', action='store_true', default=True,
                    help='allows gradient to flow through mu')
args = parser.parse_args()

batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
use_cuda = args.use_cuda
left_width = args.left_width
right_width = args.right_width
weight_dir = args.weight_dir
weight_interval = args.weight_interval
log_dir = args.log_dir
log_interval = args.log_interval
cca_reg = args.cca_reg
ledoit = cca_reg is None or cca_reg <= 0
mu_gradient = ledoit and args.mu_gradient  # True if we want to take gradient through mu


def force_mkdir(dir):
    import os
    import shutil
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)


force_mkdir(log_dir)
force_mkdir(weight_dir)
writer = SummaryWriter(log_dir)

if use_cuda:
    device = torch.device("cuda:"+str(args.gpu_num))
else:
    device = torch.device("cpu")

torch.manual_seed(args.seed)
# End of training settings

# Gathering datasets
# CREDIT: borrowed from https://gist.github.com/kdubovikov/eb2a4c3ecadd5295f68c126542e59f0a
# CREDIT:also, https://am207.github.io/2018spring/wiki/ValidationSplits.html#trainvalidation-splits-on-the-mnist-dataset
train_dataset = datasets.MNIST('mnist_data',
                               download=True,
                               train=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),  # first, convert image to PyTorch tensor
                                   transforms.Normalize((0.1307,), (0.3081,))  # normalize inputs
                               ]))

# Train/CV split
indices = list(range(len(train_dataset)))  # start with all the indices in training set
split = 10000  # define the split size

np.random.seed(0)
validation_idx = np.random.choice(indices, size=split, replace=False)
train_idx = list(set(indices) - set(validation_idx))

print(len(validation_idx), len(train_idx))

train_sampler = SubsetRandomSampler(train_idx)
validation_sampler = SubsetRandomSampler(validation_idx)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           sampler=train_sampler)
cv_loader = torch.utils.data.DataLoader(train_dataset,
                                        batch_size=batch_size,
                                        sampler=validation_sampler)

# download and transform test dataset
test_loader = torch.utils.data.DataLoader(datasets.MNIST('mnist_data',
                                                         download=True,
                                                         train=False,
                                                         transform=transforms.Compose([
                                                             transforms.ToTensor(),  # first, convert image to PyTorch tensor
                                                             transforms.Normalize((0.1307,), (0.3081,))  # normalize inputs
                                                         ])),
                                          batch_size=batch_size,
                                          shuffle=True)
# End of gathering datasets


def split_mnist(x):
    left = x[:, :, :, :14]
    right = x[:, :, :, 14:]
    return left, right


def make_model(hidden_width):  # a minimal two-layer network
    model = nn.Sequential(
        nn.Linear(14 * 28, hidden_width),
        nn.Sigmoid(),
        nn.Linear(hidden_width, hidden_width),
        nn.Sigmoid(),
        nn.Linear(hidden_width, 50)
    )
    return model


def save_model(model, niter: int, loss: float):
    import os
    torch.save(model.state_dict(), os.path.join(weight_dir, "iter-{}-loss-{}".format(niter, loss)))


class FusedModel(nn.Module):
    def __init__(self, model_left: nn.Module, model_right: nn.Module):
        super(FusedModel, self).__init__()
        self.model_left = model_left
        self.model_right = model_right

    def forward(self, data):
        left_data, right_data = split_mnist(data)
        left_data, right_data = collapse_dim(left_data), collapse_dim(right_data)
        left_out = self.model_left(left_data)
        right_out = self.model_right(right_data)
        return left_out, right_out


def collapse_dim(data):
    return torch.reshape(data, (data.shape[0], -1))


def train(epoch):
    model.train()
    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.to(device=device)
        left_out, right_out = model(data)

        loss = CorrLoss(left_out, right_out, cca_reg, ledoit, mu_gradient)
        if ledoit:
            loss, s11, mu11, s22, mu22 = loss
        loss.backward()
        optimizer.step()

        if batch_idx % weight_interval == 0:
            niter = epoch * len(train_loader) + batch_idx
            save_model(model, niter, loss.item())

        if batch_idx % log_interval == 0:
            niter = epoch * len(train_loader) + batch_idx
            quick_cv_loss = quick_validation(3)
            writer.add_scalar("CV/QuickLoss", quick_cv_loss, niter)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tCVLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader),
                   100. * batch_idx / len(train_loader), loss.item(), quick_cv_loss))
            writer.add_scalar('Train/Loss', loss.item(), niter)

            if ledoit:
                writer.add_scalar("Shrinkage11", s11.item(), niter)
                writer.add_scalar("Shrinkage22", s22.item(), niter)
                writer.add_scalar("mu11", mu11.item(), niter)
                writer.add_scalar("mu22", mu22.item(), niter)
                try:
                    writer.add_scalar("EquivCCAReg11", s11.item() * mu11.item() / (1 - s11.item()), niter)
                    writer.add_scalar("EquivCCAReg22", s22.item() * mu22.item() / (1 - s22.item()), niter)
                except ZeroDivisionError:
                    print("Seeing shrinkage of 1.0 at global step {}".format(niter))


# run a few batches to have a feeling of the cv loss
def quick_validation(num_batches):
    model.eval()
    mean_loss = 0
    for batch_idx, (data, label) in enumerate(cv_loader):
        if batch_idx >= num_batches:
            break
        data = data.to(device=device)
        left_out, right_out = model(data)
        loss = CorrLoss(left_out, right_out, cca_reg, ledoit, mu_gradient)
        if ledoit:
            loss, s11, mu11, s22, mu22 = loss
        mean_loss += loss.item()
    model.train()
    return mean_loss


def validation():  # run over the whole cv set
    return quick_validation(float("inf"))


if __name__ == "__main__":
    model_left = make_model(left_width)
    model_right = make_model(right_width)

    model = FusedModel(model_left, model_right).to(device=device)

    optimizer = optim.SGD(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',
                                  factor=0.5,
                                  patience=3,
                                  verbose=True)

    for epoch in range(epochs):
        train(epoch)
        # run validation after each epoch
        cv_loss = validation()
        writer.add_scalar("CV/Loss", cv_loss, (epoch + 1) * len(train_loader))
        scheduler.step(cv_loss)
