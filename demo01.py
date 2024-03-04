from __future__ import print_function
import os
import random
import smplx
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time

import pyrender
import trimesh

import mano



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


mano_layer = mano.load(model_path='./checkpoint/mano/', is_rhand=True, flat_hand_mean=False, use_pca=False)    # flat_hand_mean=True，这个默认是加上mean_pose了
# mano_layer = smplx.create("./checkpoint/", model_type='mano', use_pca=False)    # smplx里的mano，默认不加mean_pose
mano_layer = mano_layer.to(device)
mano_layer.eval()

mean_righthand = np.array([0.1117, -0.0429, 0.4164, 0.1088, 0.0660, 0.7562, -0.0964, 0.0909,
                            0.1885, -0.1181, -0.0509, 0.5296, -0.1437, -0.0552, 0.7049, -0.0192,
                            0.0923, 0.3379, -0.4570, 0.1963, 0.6255, -0.2147, 0.0660, 0.5069,
                            -0.3697, 0.0603, 0.0795, -0.1419, 0.0859, 0.6355, -0.3033, 0.0579,
                            0.6314, -0.1761, 0.1321, 0.3734, 0.8510, -0.2769, 0.0915, -0.4998,
                            -0.0266, -0.0529, 0.5356, -0.0460, 0.2774], dtype=np.float32)
mean_righthand = torch.FloatTensor(mean_righthand).to(device)

# 用随机数测试
x = torch.zeros([1, 45], dtype=torch.float32).to(device)
# x += mean_righthand

with torch.no_grad():
    output = mano_layer(hand_pose=x)

mesh_cam = output.vertices[0].detach().cpu().numpy()
joint_3d = output.joints[0].detach().cpu().numpy()

# pyrender可视化
vertex_colors = np.ones([mesh_cam.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]  # [10475, 4]
tri_mesh = trimesh.Trimesh(mesh_cam, mano_layer.faces,
                           vertex_colors=vertex_colors)

mesh = pyrender.Mesh.from_trimesh(tri_mesh)

scene = pyrender.Scene()
scene.add(mesh)

sm = trimesh.creation.uv_sphere(radius=0.005)
sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
tfs = np.tile(np.eye(4), (len(joint_3d), 1, 1))
tfs[:, :3, 3] = joint_3d
joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
scene.add(joints_pcl)

pyrender.Viewer(scene, use_raymond_lighting=True)


