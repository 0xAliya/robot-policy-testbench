# `scripts/01_pytorch_quickstart.py` 讲解

> 对应官方教程：<https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html>
> 运行：`uv run scripts/01_pytorch_quickstart.py`

---

## 0. 先理解四个核心概念

把训练想成"学生做题"：

| 组件 | 比喻 | 代码里 | 职责 |
|---|---|---|---|
| **Dataset**（加 DataLoader） | 题库 | `datasets.FashionMNIST(...)`、`DataLoader(batch_size=64)` | Dataset 管"第 i 条数据是什么"，返回 `(图片张量, 标签)`。DataLoader 每次取出一批（64 张），打乱顺序后交给模型。 |
| **Model** | 学生的大脑 | `NeuralNetwork(nn.Module)` | 一个带参数的函数。输入 28×28 图片，输出 10 个分数（logits），每个分数对应一类衣服。参数（权重）就是要学的东西。 |
| **loss**（损失函数） | 判卷老师 | `nn.CrossEntropyLoss()` | 比较预测和真实标签，给出一个数字表示"错得有多离谱"。越小越好。 |
| **optimizer**（优化器） | 改错方法 | `torch.optim.SGD(model.parameters(), lr=1e-3)` | 根据梯度修改模型参数。`lr`（学习率）决定每次改多大一步。 |

每一批数据的训练循环只有 5 行：

```python
pred = model(X)          # 1. 做题：前向计算出预测
loss = loss_fn(pred, y)  # 2. 判卷：算出错多少
loss.backward()          # 3. 找原因：自动算出每个参数该往哪个方向改（梯度）
optimizer.step()         # 4. 改正：按梯度更新参数
optimizer.zero_grad()    # 5. 清空梯度：PyTorch 默认会累加梯度，不清空会影响下一批
```

---

## 1. 用到的库总览

| 库 | 用到的东西 | 作用 |
|---|---|---|
| `torch` | `torch.backends.mps`、`.to(device)`、`torch.no_grad()`、`torch.save`、`torch.optim.SGD` | 核心：张量计算、GPU（MPS）、自动求梯度、优化器、保存模型 |
| `torch.nn` | `nn.Module`、`nn.Flatten`、`nn.Linear`、`nn.ReLU`、`nn.Sequential`、`nn.CrossEntropyLoss` | 搭网络的积木和损失函数 |
| `torch.utils.data` | `DataLoader` | 把数据集切成批、打乱、逐批喂出去 |
| `torchvision.datasets` | `FashionMNIST` | 现成的数据集类，自动下载和读取 |
| `torchvision.transforms` | `ToTensor` | 把图片转成张量 |

脚本流程：

```
准备数据 → 选设备 → 定义模型 → 定义 loss + optimizer → 训练/测试循环 → 保存模型
```

---

## 2. 第 1 步：准备数据（Dataset + DataLoader）

```python
training_data = datasets.FashionMNIST(root="data", train=True,  download=True, transform=ToTensor())
test_data     = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())
```

- `datasets.FashionMNIST`（torchvision）：衣服图片数据集，10 类（T 恤、裤子、鞋……），每张 28×28 灰度图。
  - `root="data"`：下载到 `./data/`（运行时刷屏的百分比就是在下载）
  - `train=True` 取 6 万张训练集，`train=False` 取 1 万张测试集
  - `transform=ToTensor()`：每次取图时，把原始图片（PIL 格式，像素 0~255）转成形状 `[1, 28, 28]`、取值 0~1 的浮点张量
- 取出第 i 条：`training_data[i]` 返回 `(图片张量, 标签整数)`

```python
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader  = DataLoader(test_data, batch_size=64)
```

- `DataLoader`（torch.utils.data）：每次打包 64 张一起给模型，比一张一张处理快得多
- `shuffle=True`：每个 epoch 打乱训练集顺序，防止模型记住顺序；测试集不需要打乱

```python
for X, y in test_dataloader:
    print(X.shape)   # [64, 1, 28, 28] → N(批大小), C(通道), H(高), W(宽)
    print(y.shape)   # [64]，int64 类型的标签
    break
```

只取一批看形状，确认数据没问题，`break` 后不再继续。

---

## 3. 第 2 步：选设备

```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
```

- `torch.backends.mps`：MPS 是 Apple 芯片的 GPU 加速后端，相当于 Mac 上的 CUDA
- 后面模型和数据都要 `.to(device)` 放到**同一个设备**上，否则报错

---

## 4. 第 3 步：定义模型（Model）

```python
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512), nn.ReLU(),
            nn.Linear(512, 512),   nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)
```

| 组件 | 做什么 | 形状变化 |
|---|---|---|
| `nn.Module` | 所有网络的基类，继承后 PyTorch 自动登记里面的参数 | — |
| `nn.Flatten()` | 把每张图拉平成一行 | `[64,1,28,28]` → `[64,784]` |
| `nn.Linear(784, 512)` | 全连接层 `y = xW + b`，W 和 b 是要学的参数 | `[64,784]` → `[64,512]` |
| `nn.ReLU()` | 激活函数 `max(0, x)`，提供非线性；没有它多层 Linear 等价于一层 | 不变 |
| `nn.Sequential(...)` | 把几层串起来按顺序执行 | — |
| 最后 `Linear(512, 10)` | 输出 10 个分数（logits），最高分就是预测类别 | `[64,512]` → `[64,10]` |

- `forward()`：规定数据怎么流过网络；调用 `model(X)` 时自动执行它
- `model = NeuralNetwork().to(device)`：把参数放到 MPS 上
- `print(model)`：打印网络结构

---

## 5. 第 4 步：损失函数 + 优化器

```python
loss_fn   = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
```

- `nn.CrossEntropyLoss`：分类任务的标准损失。内部先 softmax 把 10 个分数变成概率，再看正确类别的概率有多低，越低惩罚越大
- `torch.optim.SGD`：随机梯度下降，规则是 `参数 -= lr × 梯度`
  - `model.parameters()`：告诉优化器要改哪些参数（三层 Linear 的 W 和 b，共约 67 万个数）
  - `lr=1e-3`：学习率，每次改动的步长

---

## 6. 第 5 步：训练函数和测试函数

### `train()`：跑完一遍训练集（1 个 epoch）

```python
model.train()                          # 切换到训练模式
for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)  # 数据搬到 MPS
    pred = model(X)                    # ① 前向：得到 [64,10] 的预测
    loss = loss_fn(pred, y)            # ② 算损失：一个标量
    loss.backward()                    # ③ 反向传播：autograd 自动算出每个参数的梯度
    optimizer.step()                   # ④ 用梯度更新参数
    optimizer.zero_grad()              # ⑤ 清空梯度（PyTorch 默认累加梯度）
    if batch % 100 == 0:
        print(loss.item(), ...)        # .item() 把单元素张量转成 Python 数字
```

6 万张 ÷ 64 ≈ 938 批，每 100 批打印一次 loss。

### `test()`：在测试集上评估，不更新参数

```python
model.eval()                 # 切换到评估模式
with torch.no_grad():        # 不记录计算图，省内存、更快
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        test_loss += loss_fn(pred, y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
```

- `pred.argmax(1)`：在 10 个分数里取最大值的下标 = 预测类别
- `== y`：和真实标签逐个比较得到 True/False，转浮点求和 = 这一批猜对的张数
- 最后打印准确率和平均 loss

> `model.train()` / `model.eval()` 在这个网络里其实没有效果，它们只影响 Dropout、BatchNorm 这类层。但要养成习惯，以后的网络会用到。

---

## 7. 第 6 步：主循环 + 保存

```python
for t in range(2):          # 2 个 epoch
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

torch.save(model.state_dict(), "model.pth")
```

- 每训练完一个 epoch 测试一次。实际运行结果：

  | Epoch | 测试准确率 | 平均 loss |
  |---|---|---|
  | 1 | 44.7% | 2.157 |
  | 2 | 51.8% | 1.872 |

  官方用 SGD、lr=1e-3，大约 5 个 epoch 到 64%，所以 2 个 epoch 到 51.8% 是正常的。
- `model.state_dict()`：一个字典，存所有参数张量（`层名 → 权重`）
- `torch.save`：存成 `model.pth`
- 以后加载：

  ```python
  model = NeuralNetwork().to(device)
  model.load_state_dict(torch.load("model.pth"))
  ```

---

## 8. 一句话串起来

**torchvision** 准备数据 → **DataLoader** 分批 → **nn** 搭网络、算 loss → **autograd**（`loss.backward()`）求梯度 → **optim** 改参数 → 循环 2 个 epoch → **`torch.save`** 存下来。

---

## 9. 和机器人策略的对应关系

| 这个脚本 | 以后做 RL（如 Reacher-v5） |
|---|---|
| Model：图片 → 10 个分数 | 策略网络：10 维观测 → 2 维动作 |
| Dataset：FashionMNIST | 和环境交互收集到的轨迹 |
| loss：CrossEntropyLoss | PPO / SAC 的目标函数 |
| optimizer：SGD | 一般用 Adam |

那 5 行训练循环的结构基本不变，stable-baselines3 在内部帮你写好了。

---

## 10. 自测问答

### Q1：训练循环的四步是什么？

对每一批数据 `(X, y)` 重复下面四步：

| 步骤 | 代码 | 做什么 |
|---|---|---|
| ① 前向传播 | `pred = model(X)` | 让模型用当前参数做预测 |
| ② 计算损失 | `loss = loss_fn(pred, y)` | 用一个数字衡量预测和真实标签差多少 |
| ③ 反向传播 | `loss.backward()` | autograd 自动算出每个参数的梯度，也就是"往哪个方向改能让 loss 变小" |
| ④ 更新参数 | `optimizer.step()` | 按梯度把参数改一小步（SGD：`参数 -= lr × 梯度`） |

还有一行配套操作：`optimizer.zero_grad()`，用来**清空梯度**。PyTorch 默认把梯度累加起来，不清空的话，这一批的梯度会和上一批混在一起。所以前面说"5 行"，就是这四步加上清零。清零放在 ③ 之前或 ④ 之后都可以，只要每批清一次。

口诀：**预测 → 算错 → 找原因 → 改正**。

### Q2：loss 和 accuracy 有什么区别？

| | loss（损失） | accuracy（准确率） |
|---|---|---|
| 是什么 | 预测概率和真实标签之间的"距离" | 猜对的样本占总数的比例 |
| 取值 | 0 到正无穷，**越小越好** | 0% 到 100%，**越大越好** |
| 给谁用 | 给**优化器**用：它是可导的，能算梯度 | 给**人**看：直观好懂 |
| 灵敏度 | 连续变化。正确类别的概率从 0.4 升到 0.6，loss 会下降 | 只看 argmax 对不对。0.4 和 0.6 只要都是最高分，准确率就一样 |

为什么不直接拿准确率去训练？因为 `argmax` 不可导：把参数稍微改一点，猜对的张数通常不变，梯度几乎处处为 0，优化器不知道该往哪边改。CrossEntropy 是平滑的，参数每改一点它都会跟着变，所以能用来训练。

本次运行的例子：第 2 个 epoch 的 loss 从 2.157 降到 1.872，accuracy 从 44.7% 升到 51.8%。两者通常一起变好，但不总是严格同步。

### Q3：为什么要分训练集和测试集？

**为了检验模型是"真学会了"还是"背答案"。**

- 训练集（6 万张）：模型用它更新参数，相当于平时做的练习题。
- 测试集（1 万张）：模型**从没见过、也从没用它更新参数**，相当于期末考试。

如果只看训练集上的表现，模型可能把训练图片死记下来，训练准确率很高，换一张新图就不行了。这叫**过拟合**。只有在没见过的数据上表现好，才说明模型学到了"T 恤长什么样"这种可以推广的规律，也就是**泛化能力**。

所以代码里：

- `test()` 里用 `torch.no_grad()`，而且没有 `backward()` 和 `step()`，保证测试数据不会影响参数。
- 测试集不 shuffle，因为只是评估，顺序不影响结果。

> 更严格的做法会再分出一个**验证集**，用来调学习率、选 epoch 数。测试集只在最后用一次，否则调参时就等于间接"偷看"了考题。
>
> 对应到机器人策略：在训练用的随机种子或场景上表现好不算数，要在**没见过的初始状态、物体位置、甚至新环境**上测成功率，这正是 testbench 要做的事。
