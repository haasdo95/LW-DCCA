"""
This file contains all that is needed to run dcca on the left and right half of mnist numbers
"""

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms

from dcca_loss import CorrLoss

batch_size = 10
use_cuda = True
left_width = 2038
right_width = 1608
cca_reg = 0.001
ledoit = cca_reg <= 0
mu_gradient = True  # True if we want to take gradient through mu

if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# CREDIT: borrowed from https://gist.github.com/kdubovikov/eb2a4c3ecadd5295f68c126542e59f0a
train_loader = torch.utils.data.DataLoader(datasets.MNIST('mnist_data',
                                                          download=True,
                                                          train=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),  # first, convert image to PyTorch tensor
                                                              transforms.Normalize((0.1307,), (0.3081,))  # normalize inputs
                                                          ])),
                                           batch_size=batch_size,
                                           shuffle=True)

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


def split_mnist(x):
    left = x[:, :, :, :14]
    right = x[:, :, :, 14:]
    return left, right


def make_model(hidden_width):  # a minimal two-layer network
    model = nn.Sequential(
        nn.Linear(14 * 28, hidden_width),
        nn.ReLU(),
        nn.Linear(hidden_width, hidden_width),
        nn.ReLU(),
        nn.Linear(hidden_width, 50)
    )
    return model


class FusedModel(nn.Module):
    def __init__(self, model_left: nn.Module, model_right: nn.Module):
        super(FusedModel, self).__init__()
        self.model_left = model_left
        self.model_right = model_right

    def forward(self, data):
        assert data.shape == (batch_size, 1, 28, 28)
        left_data, right_data = split_mnist(data)
        left_data, right_data = collapse_dim(left_data), collapse_dim(right_data)
        left_out = self.model_left(left_data)
        right_out = self.model_right(right_data)
        return left_out, right_out


def collapse_dim(data):
    return torch.reshape(data, (data.shape[0], -1))


if __name__ == "__main__":
    model_left = make_model(left_width)
    model_right = make_model(right_width)

    model = FusedModel(model_left, model_right)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for batch_id, (data, label) in enumerate(train_loader):
        data = data.to(device=device)
        left_out, right_out = model(data)
        loss = CorrLoss(left_out, right_out)
        print(loss)
        break
