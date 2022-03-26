from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from tinydl import nn
import tinydl
import numpy as np
import time


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2D(in_channels, out_channels // 2, kernel_size=3, padding=1, stride=strides)
        self.bn1 = nn.BatchNorm(out_channels//2)
        self.conv2 = nn.Conv2D(out_channels//2, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm(out_channels)
        if strides == 2:
            self.shortcut = nn.Conv2D(in_channels, out_channels, kernel_size=1, padding=0, stride=2)
        else:
            self.shortcut = None

    def forward(self, data):
        x = tinydl.relu(self.bn1(self.conv1(data)))
        x = tinydl.relu(self.bn2(self.conv2(x)))
        if self.shortcut is not None:
            data = self.shortcut(data)
        return data + x
        

class TinydlNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1*28*28
        self.stage1_block1 = ResBlock(1, 16, 2)
        self.stage1_block2 = ResBlock(16, 16, 1)
        self.stage1_block3 = ResBlock(16, 16, 1)
        
        # 16*14*14
        self.stage2_block1 = ResBlock(16, 32, 2)
        self.stage2_block2 = ResBlock(32, 32, 1)
        self.stage2_block3 = ResBlock(32, 32, 1)
        
        # 32*7*7
        self.fc = nn.Linear(32*7*7, 10)

    def forward(self, x):
        x = self.stage1_block1(x)
        x = self.stage1_block2(x)
        x = self.stage1_block3(x)
        
        x = self.stage2_block1(x)
        x = self.stage2_block2(x)
        x = self.stage2_block3(x)
        
        x = x.view(x.shape()[0], 32*7*7)
        x = self.fc(x)
        return x


def loss_fn(pred, y):
    m = pred.mean(1, keep_dim=True)
    p = pred - m
    p = p.exp()
    s = p.sum(1, keep_dim=True)
    p = p / s
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

        if batch % 10 == 0:
            time_used += time.time()
            speed = time_used / 10
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


if __name__ == "__main__":
    main()
