from UNet import UNet
import torch, os
from time import time
from tqdm import tqdm


import torch
import time

iterations = 100   # 重复计算的轮次

model = UNet().eval()
device = torch.device("cuda:0")
model.to(device)

random_input1 = torch.randn(1, 3, 384, 384).to(device)

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

# GPU预热
for _ in range(50):
    _ = model(random_input1)

# 测速
times = 0    # 存储每轮iteration的时间
with torch.no_grad():
    for iter in range(iterations):
        start = time.time()
        model(random_input1)
        times += time.time()- start
        # print(curr_time)

print("Inference time: {:.6f}, FPS: {} ".format(times, 100/times *1))
