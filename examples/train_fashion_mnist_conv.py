from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from tinydl import nn
import tinydl
import numpy as np
import time
import pickle


class TinydlNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1x28x28
        self.conv1 = nn.Conv2D(1, 16, kernel_size=3, padding=1, stride=2)
        self.bn = nn.BatchNorm(16)
        # 16*14*14
        self.conv2 = nn.Conv2D(16, 32, kernel_size=3, padding=1, stride=2)
        # 32*7*7
        self.fc = nn.Linear(32*7*7, 10)

    def forward(self, x):
        x = tinydl.relu(self.bn(self.conv1(x)))
        x = tinydl.relu(self.conv2(x))
        x = x.view(x.shape()[0], 32*7*7)
        x = self.fc(x)
        return x


def loss_fn(pred, y):
    m = pred.mean(1, keep_dim=True)
    p = pred - m
    p = p.exp()
    s = p.sum(1, keep_dim=True)
    p = p / s + tinydl.Tensor([[1e-45]])
    neg = tinydl.Tensor([[-1]])
    idx = tinydl.Tensor(np.arange(10)).view(1, 10)
    y = y.view(y.shape()[0], 1)
    mask = tinydl.equal(idx, y)
    
    loss = tinydl.log(p) * neg
    loss = (loss * mask).sum(1)
    return loss.mean(0)


def train(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader.dataset)
    model.train()
    time_used = - time.time()
    for batch, (X, y) in enumerate(dataloader):
        X = X.numpy()
        y = y.numpy()
        # X = X.reshape(X.shape[0], -1)
        X = tinydl.Tensor(X)
        y = tinydl.Tensor(y)

        # Compute prediction error
        # print("=====", X.shape())
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward(tinydl.Tensor([1]))

        optimizer.step()
        optimizer.zero_grad()

        if batch % 50 == 0:
            time_used += time.time()
            speed = time_used / 50
            time_used = - time.time()
            loss, current = loss.numpy().item(), batch * X.shape()[0]
            print(f"[epoch: {epoch}] loss: {loss:>7f}  [{current:>5d}/{size:>5d}] [{speed:>.3f} s/b]")


def test(dataloader, model, loss_fn, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    model.eval()
    for X, y in dataloader:
        X = X.numpy()
        y = y.numpy()
        # X = X.reshape(X.shape[0], -1)
        X = tinydl.Tensor(X)
        y1 = tinydl.Tensor(y)
        
        pred = model(X)

        test_loss += loss_fn(pred, y1).numpy().item()
        correct += (pred.numpy().argmax(1) == y).sum()

    test_loss /= num_batches
    correct /= size
    print(f"[epoch: {epoch}] Test Error: \n\tAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    training_data = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
    )

    test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor()
    )

    batch_size = 64
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    model_tinydl = TinydlNet()
    opt = tinydl.optim.SGD(1e-3)
    opt.attach(model_tinydl.named_parameters())

    for epoch in range(50):
        if epoch == 40:
            opt.set_lr(1e-4)
        train(train_dataloader, model_tinydl, loss_fn, opt, epoch)
        test(test_dataloader, model_tinydl, loss_fn, epoch)

        state = model_tinydl.state_dict()
        with open("ckpt_conv.pkl", "wb") as f:
            pickle.dump(state, f)


if __name__ == "__main__":
    main()
