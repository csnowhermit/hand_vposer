import os
import numpy as np
import torch
import torch.nn as nn

from hand_poser import HandPoser

poser = HandPoser(num_neurons=512, latentD=10)

# x = torch.randn([1024, 45], dtype=torch.float32)

data = np.load("./data/InterHand2.6M_label_hand_pose_train.npy")[0: 2, :].reshape(2, -1)
x = torch.tensor(data, dtype=torch.float32)
print(x.shape)


loss_func = nn.L1Loss()

for epoch in range(5000000):
    pred = poser(x)

    loss = loss_func(pred, x)
    loss.backward()

    if epoch % 10 == 0:
        print("Epoch: %d, Loss: %.06f" % (epoch, loss.item()))

# 推理时候


