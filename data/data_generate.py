import numpy as np
import torch

data = []
for _ in range(5):
    yaw = np.random.uniform(-np.pi, np.pi)
    x = np.random.uniform(-0.5, 0.5)
    y = np.random.uniform(-0.5, 0.5)
    thetas = np.random.uniform(-np.pi, np.pi, 4)
    tensor = [yaw, x, y] + list(thetas)
    # tensor[7] = np.pi/2
    # tensor[8] = np.pi/2

    tensor = [round(val, 3) for val in tensor] # 保留3位小数

    data.append(tensor)

data_tensor = torch.FloatTensor(data)
print(data_tensor)