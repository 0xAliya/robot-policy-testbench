"""照 PyTorch 官方 Quickstart：FashionMNIST 分类，训练 2 个 epoch，用 MPS。
https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

整体流程：准备数据 → 选设备 → 定义模型 → 定义 loss + optimizer → 训练/测试循环 → 保存模型
详细讲解见 docs/01_pytorch_quickstart_讲解.md
"""
import torch                                # 核心库：张量、自动求梯度、设备管理、保存模型
from torch import nn                        # 神经网络积木：层、激活函数、损失函数
from torch.utils.data import DataLoader     # 把数据集切成一批一批地喂给模型
from torchvision import datasets            # 现成的数据集（FashionMNIST、CIFAR10 等）
from torchvision.transforms import ToTensor # 把图片转成张量

# ---------- 1. Dataset + DataLoader ----------
# FashionMNIST：10 类衣服的 28×28 灰度图，训练集 6 万张，测试集 1 万张
# root="data"    → 下载/读取到 ./data/ 目录（已被 .gitignore 忽略）
# train=True     → 训练集；train=False → 测试集
# download=True  → 本地没有就自动下载
# transform      → 每次取图时做的处理：ToTensor 把 PIL 图片（0~255）变成 [1, 28, 28] 的浮点张量（0~1）
training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

# 每批 64 张。批越大，每步越稳但越吃内存
batch_size = 64
# shuffle=True：每个 epoch 打乱训练集，防止模型记住顺序；测试集只做评估，不用打乱
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# 只取一批看看形状，确认数据没问题
# X: [64, 1, 28, 28] → N(批大小), C(通道，灰度图为 1), H(高), W(宽)
# y: [64]，每个元素是 0~9 的类别编号（int64）
for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# ---------- 2. Model ----------
# MPS = Apple 芯片的 GPU 加速（相当于 Mac 上的 CUDA）；不可用就退回 CPU
# 注意：模型和数据必须放在同一个 device 上，否则会报错
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")


# 自定义网络都要继承 nn.Module，这样 PyTorch 才能自动找到里面所有可训练参数
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 把 [N, 1, 28, 28] 拉平成 [N, 784]，全连接层只接受一维特征
        self.flatten = nn.Flatten()
        # Sequential：按顺序把各层串起来
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),  # 全连接层 y = xW + b：[N, 784] → [N, 512]
            nn.ReLU(),                # 激活函数 max(0, x)，提供非线性；没有它多层 Linear 等价于一层
            nn.Linear(512, 512),      # [N, 512] → [N, 512]
            nn.ReLU(),
            nn.Linear(512, 10),       # [N, 512] → [N, 10]：10 个类别各一个分数（logits）
        )

    # forward 规定数据怎么流过网络；调用 model(X) 时会自动执行它，不要直接调 forward
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)  # logits：未归一化的分数，最高分即预测类别
        return logits


# .to(device)：把模型参数搬到 MPS 上
model = NeuralNetwork().to(device)
print(model)

# ---------- 3. loss + optimizer ----------
# 交叉熵损失：分类任务标准选择。内部先 softmax 把 logits 变成概率，
# 再看正确类别的概率有多低——越低惩罚越大。所以模型输出 logits 即可，不用自己加 softmax
loss_fn = nn.CrossEntropyLoss()
# SGD 随机梯度下降：参数 -= lr × 梯度
# model.parameters()：告诉优化器要更新哪些参数（3 层 Linear 的 W 和 b，约 67 万个数）
# lr=1e-3：学习率（步长）。太大会震荡发散，太小学得慢
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    """把训练集完整过一遍（1 个 epoch），边看边更新参数。"""
    size = len(dataloader.dataset)  # 训练集总样本数：60000
    model.train()                   # 训练模式（影响 Dropout/BatchNorm 等层；本网络没有，但要养成习惯）
    # 每次循环拿到一批：X 是 64 张图，y 是对应的 64 个标签；batch 是批序号
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)  # 数据也要搬到和模型相同的设备

        # ===== 训练循环四步 =====
        pred = model(X)            # ① 前向：用当前参数做预测，得到 [64, 10]
        loss = loss_fn(pred, y)    # ② 算损失：预测和标签差多少，一个标量

        loss.backward()            # ③ 反向传播：autograd 自动算出每个参数的梯度
        optimizer.step()           # ④ 更新：按梯度把参数改一小步
        optimizer.zero_grad()      # 清空梯度：PyTorch 默认累加梯度，不清会和下一批混在一起

        # 每 100 批打印一次进度（60000 / 64 ≈ 938 批）
        if batch % 100 == 0:
            # .item()：把只有一个元素的张量转成普通 Python 数字
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    """在测试集上评估：只预测、不更新参数，看模型在没见过的数据上表现如何。"""
    size = len(dataloader.dataset)  # 测试集总样本数：10000
    num_batches = len(dataloader)   # 批数：10000 / 64 ≈ 157
    model.eval()                    # 评估模式
    test_loss, correct = 0, 0
    # no_grad：不记录计算图、不算梯度 → 省内存、更快；测试时本来也不需要梯度
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # pred.argmax(1)：每行 10 个分数里取最大值的下标 → 预测类别
            # == y 得到 True/False，转成 1.0/0.0 后求和 → 这一批猜对的张数
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches  # 平均每批的 loss
    correct /= size           # 准确率 = 猜对总数 / 样本总数
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# ---------- 4. 训练 ----------
# epoch：把整个训练集完整过一遍。这里过 2 遍，每遍结束后测试一次，观察准确率变化
epochs = 2
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

# state_dict()：一个字典 {层名: 参数张量}，只存参数不存代码
# 以后加载：model = NeuralNetwork(); model.load_state_dict(torch.load("model.pth"))
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
