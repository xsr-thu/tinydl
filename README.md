# TinyDL

TinyDL 是一个用C++/CUDA实现底层计算和调度，并提供python API接口的，一个简单的深度学习框架。

目前其还在快速开发中。暂时支持简单的MLP模型，Conv结构会尽快支持。

该项目已学习，验证为主，目前仅仅依赖CUDA和pybind11，大部分的kernel均为手写。后续对于Conv等算子会使用CUDNN

# 安装
1.Clone Repo并更新子模块
```
git clone https://github.com/xsr-thu/tinydl.git
cd tinydl
git submodule update --init
```

2.编译底层的C++/CUDA实现
```
mkdir build
cd build
cmake ..
make
```

3.安装python库
```
python3 setup.py install
# or
python3 setup.py develop
```

# 开发进展
## 算子支持
- [X] add 
- [X] sub
- [X] mul
- [X] div
- [X] equal
- [X] less then
- [X] less equal
- [X] greater then
- [X] greater equal
- [X] exp
- [X] log
- [X] sigmoid
- [X] relu
- [X] matmul
- [X] view
- [X] reduce sum
- [X] reduce mean
- [X] 自动 broadcast / reduction
- [X] Conv
- [X] BN
- [X] min / max (forward)
- [ ] argmax/argmin
- [ ] indexing
- [ ] cumsum
- [ ] pooling
## 数据类型
- [X] float32
- [ ] int/uint32/uint54
- [ ] bool
## 计算范式
- [X] imperative
- [ ] static graph
## 特性
- [ ] 异步执行流
- [ ] 多卡
- [ ] 混合精度

# Example
参考examples目录下的 train_fashion_mnist.py

其训练log见examples目录下的 worklog.txt。
```
[epoch: 0] loss: 2.584504  [    0/60000]
[epoch: 0] loss: 2.186415  [ 3200/60000]
[epoch: 0] loss: 1.994072  [ 6400/60000]
[epoch: 0] loss: 1.859542  [ 9600/60000]
[epoch: 0] loss: 1.663692  [12800/60000]
[epoch: 0] loss: 1.664524  [16000/60000]
[epoch: 0] loss: 1.577286  [19200/60000]
[epoch: 0] loss: 1.392085  [22400/60000]
[epoch: 0] loss: 1.296988  [25600/60000]
[epoch: 0] loss: 1.308505  [28800/60000]
[epoch: 0] loss: 1.240550  [32000/60000]
[epoch: 0] loss: 1.066875  [35200/60000]
[epoch: 0] loss: 1.229970  [38400/60000]
[epoch: 0] loss: 1.210799  [41600/60000]
[epoch: 0] loss: 1.068198  [44800/60000]
[epoch: 0] loss: 0.954115  [48000/60000]
[epoch: 0] loss: 1.098883  [51200/60000]
[epoch: 0] loss: 1.058162  [54400/60000]
[epoch: 0] loss: 0.994515  [57600/60000]
[epoch: 0] Test Error:
        Accuracy: 69.6%, Avg loss: 1.007727

[epoch: 1] loss: 0.984413  [    0/60000]
[epoch: 1] loss: 1.050318  [ 3200/60000]
[epoch: 1] loss: 1.065541  [ 6400/60000]
[epoch: 1] loss: 1.065259  [ 9600/60000]
[epoch: 1] loss: 0.817950  [12800/60000]
[epoch: 1] loss: 0.981624  [16000/60000]
[epoch: 1] loss: 1.001392  [19200/60000]
[epoch: 1] loss: 0.771233  [22400/60000]
[epoch: 1] loss: 0.830159  [25600/60000]
[epoch: 1] loss: 0.924094  [28800/60000]
[epoch: 1] loss: 0.853215  [32000/60000]
[epoch: 1] loss: 0.655708  [35200/60000]
[epoch: 1] loss: 0.907100  [38400/60000]
[epoch: 1] loss: 0.923296  [41600/60000]
[epoch: 1] loss: 0.824613  [44800/60000]
[epoch: 1] loss: 0.666591  [48000/60000]
[epoch: 1] loss: 0.852497  [51200/60000]
[epoch: 1] loss: 0.859955  [54400/60000]
[epoch: 1] loss: 0.803036  [57600/60000]
[epoch: 1] Test Error:
        Accuracy: 74.1%, Avg loss: 0.804829

......

[epoch: 49] loss: 0.412116  [38400/60000]
[epoch: 49] loss: 0.510278  [41600/60000]
[epoch: 49] loss: 0.564337  [44800/60000]
[epoch: 49] loss: 0.395831  [48000/60000]
[epoch: 49] loss: 0.546788  [51200/60000]
[epoch: 49] loss: 0.443715  [54400/60000]
[epoch: 49] loss: 0.361649  [57600/60000]
[epoch: 49] Test Error:
        Accuracy: 84.5%, Avg loss: 0.432874
```

该参考仅仅证明其能在FashionMNIST数据集上快速收敛，并没有调参到最佳性能。
