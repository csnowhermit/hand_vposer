import os
import numpy as np
import torch
import torch.nn as nn
import mano

MANO_JOINT_NAME = [
    'wrist',
    'index1',
    'index2',
    'index3',    # 数字越大越靠近指尖
    'middle1',
    'middle2',
    'middle3',
    'pinky1',
    'pinky2',
    'pinky3',
    'ring1',
    'ring2',
    'ring3',
    # 'thumb1',    # 这个点的2d坐标没在手上，也不知道在哪，直接用误差太大
    'thumb2',
    'thumb3',
    'thumb',    # 指尖
    'index',
    'middle',
    'ring',
    'pinky',
]

# 标签中3d关键点的joint_name顺序（这里joint_name是mano中的关节名）
label_joint_name = ['thumb',
                    'thumb3',    # 数字越大越靠近指尖 标签中的3d关键点是从指尖开始
                    'thumb2',
                    'thumb1',
                    'index',
                    'index3',
                    'index2',
                    'index1',
                    'middle',
                    'middle3',
                    'middle2',
                    'middle1',
                    'ring',
                    'ring3',
                    'ring2',
                    'ring1',
                    'pinky',
                    'pinky3',
                    'pinky2',
                    'pinky1',
                    'wrist']

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


common_loss = nn.SmoothL1Loss()

# 计算重投影损失用到的点
proj_joint_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20]


mean_righthand = np.array([0.1117, -0.0429, 0.4164, 0.1088, 0.0660, 0.7562, -0.0964, 0.0909,
                            0.1885, -0.1181, -0.0509, 0.5296, -0.1437, -0.0552, 0.7049, -0.0192,
                            0.0923, 0.3379, -0.4570, 0.1963, 0.6255, -0.2147, 0.0660, 0.5069,
                            -0.3697, 0.0603, 0.0795, -0.1419, 0.0859, 0.6355, -0.3033, 0.0579,
                            0.6314, -0.1761, 0.1321, 0.3734, 0.8510, -0.2769, 0.0915, -0.4998,
                            -0.0266, -0.0529, 0.5356, -0.0460, 0.2774], dtype=np.float32)


if __name__ == '__main__':
    batch_size = 1024

    # with torch.no_grad():
    #     output = mano_layer(betas=torch.randn([batch_size, 10], dtype=torch.float32).to(device),  # [1, 10]
    #                         hand_pose=torch.randn([batch_size, 45], dtype=torch.float32).to(device),
    #                         global_orient=torch.randn([batch_size, 3], dtype=torch.float32).to(device),
    #                         transl=torch.randn([batch_size, 3], dtype=torch.float32).to(device))  # [1, 3]
    #     joint3d = output.joints.cpu().numpy()
    #     print(joint3d.shape)
    print(len(proj_joint_list))
