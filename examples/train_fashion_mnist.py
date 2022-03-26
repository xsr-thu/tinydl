from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from tinydl import nn
import tinydl
import numpy as np
import time


class TinydlNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = tinydl.relu(self.fc1(x))
        x = tinydl.relu(self.fc2(x))
        x = self.fc3(x)
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
        X = X.reshape(X.shape[0], -1)
        X = tinydl.Tensor(X)
        y = tinydl.Tensor(y)

        # Compute prediction error
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
        X = X.reshape(X.shape[0], -1)
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
