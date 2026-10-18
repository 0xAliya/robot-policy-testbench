# robot-policy-testbench

> 这是一个**个人学习项目**：从零学习 PyTorch，逐步过渡到用强化学习 / 模仿学习训练和评测机器人策略（MuJoCo 仿真环境）。代码以理解为主，不追求性能和工程完整性。

## 运行设备

| 项目 | 配置 |
|---|---|
| 机器 | MacBook，Apple M3 Max（30 核 GPU） |
| 内存 | 36 GB 统一内存 |
| 系统 | macOS 15.6.1 |
| 加速 | PyTorch **MPS** 后端（Apple GPU），没有 CUDA |

脚本会自动选择设备：`mps` 可用就用 `mps`，否则退回 `cpu`。在 Linux / NVIDIA 机器上运行时，需要把 device 改成 `cuda`。

## 环境

- Python 3.12（用 [uv](https://docs.astral.sh/uv/) 管理虚拟环境）
- torch 2.14.1 / torchvision 0.29.1
- gymnasium 1.4.0 + mujoco 3.16.0
- stable-baselines3 2.9.0
- imageio、matplotlib

```bash
uv sync                               # 按 uv.lock 安装依赖
uv run scripts/01_pytorch_quickstart.py
```

## 目录

| 路径 | 内容 |
|---|---|
| `scripts/01_pytorch_quickstart.py` | PyTorch 官方 Quickstart：FashionMNIST 分类，在 MPS 上训练 2 个 epoch |
| `docs/01_pytorch_quickstart_讲解.md` | 01 脚本的逐行讲解和自测问答 |

