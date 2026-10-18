"""加载训练好的 SAC 模型，在 Reacher-v5 上录 3 段回合，存成 results/reacher.gif。"""
from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
from stable_baselines3 import SAC

ENV_ID = "Reacher-v5"
MODEL_PATH = Path("models/sac_reacher.zip")
GIF_PATH = Path("results/reacher.gif")
N_EPISODES = 3
FRAME_STRIDE = 2   # 环境 50 fps，每 2 帧取 1 帧 → GIF 25 fps，播放速度和真实一致
SIZE = 320         # 画面边长（像素）
PAUSE_FRAMES = 12  # 每段回合结束后停留约 0.5 秒

GIF_PATH.parent.mkdir(exist_ok=True)

# render_mode="rgb_array"：env.render() 返回画面的像素数组 [H, W, 3]，而不是弹出窗口
# 默认相机离得太远、斜着看，手臂和目标点很小；这里改成正上方俯视、拉近到刚好看全场地
camera = dict(distance=0.8, elevation=-90.0, azimuth=90.0, lookat=np.array([0.0, 0.0, 0.0]))
env = gym.make(ENV_ID, render_mode="rgb_array", width=SIZE, height=SIZE, default_camera_config=camera)
model = SAC.load(MODEL_PATH, device="cpu")

frames = []
for ep in range(N_EPISODES):
    obs, info = env.reset(seed=100 + ep)  # 每段用不同种子 → 目标点位置不同
    ep_return, t, done = 0.0, 0, False
    while not done:
        if t % FRAME_STRIDE == 0:
            frames.append(env.render())
        # deterministic=True：直接用策略输出的均值动作，不加探索噪声
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += reward
        t += 1
        done = terminated or truncated  # Reacher 不会 terminated，50 步后 truncated
    last = env.render()
    frames.extend([last] * PAUSE_FRAMES)
    # obs[8:10] 是 指尖 - 目标 的 (x, y) 向量，它的长度就是最终距离
    dist = np.linalg.norm(obs[8:10])
    print(f"episode {ep+1}: {t} steps, return {ep_return:7.2f}, final distance {dist*100:.1f} cm")

env.close()
# duration 单位是毫秒；loop=0 表示无限循环
imageio.mimsave(GIF_PATH, frames, duration=1000 / 25, loop=0)
print(f"Saved {len(frames)} frames to {GIF_PATH} ({GIF_PATH.stat().st_size/1e6:.1f} MB)")
