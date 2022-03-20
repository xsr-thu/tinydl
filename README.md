# TinyDL

TinyDL 是一个用C++/CUDA实现底层计算和调度，并提供python API接口的，一个简单的深度学习框架。

目前其还在快速开发中。暂时支持简单的MLP模型，Conv结构会尽快支持。

该项目已学习，验证为主，目前仅仅依赖CUDA和pybind11，大部分的kernel均为手写。后续对于Conv等算子会使用CUDNN

# 安装
1.编译底层的C++/CUDA实现
```
mkdir build
cd build
cmake ..
make
```

2.安装python库
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
- [ ] Conv
- [ ] BN
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

该参考仅仅证明其能在FashionMNIST数据集上快速收敛，并没有调参到最佳性能。
