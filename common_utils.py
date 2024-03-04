import os
import torch
import numpy as np

import config


'''
    数据集方差读取，同时降维，输出降维算子和主成分特征值的逆
'''
def loadcovInv(covPath, down2dim = 10):
    covMatrix = np.load(covPath)
    eig_value, eig_vector = np.linalg.eig(covMatrix)
    argSor = np.argsort(eig_value)
    dimdownSor =  argSor[:-down2dim-1:-1]
    dimdownOpt = eig_vector[dimdownSor]
    invprincleValue = np.diag(1/eig_value[dimdownSor])
    return dimdownOpt, invprincleValue

'''
    降维后的马氏距离
'''
def hand_prior_Mdistance(pose1, pose2, dimdownOpt, eign_diag):
    pose1_dimdown = pose1@dimdownOpt.transpose(0,1)
    pose1_dimdown.to(config.device)
    pose2_dimdown = pose2@dimdownOpt.transpose(0,1)
    return torch.trace((pose1_dimdown - pose2_dimdown)@eign_diag@(pose1_dimdown - pose2_dimdown).T)/pose1.numel()



def Loss_Proj2D(output, betas, transl, kp2d, cam):
    batch_size = output.shape[0]

    mano_pred = config.mano_layer(betas=betas,  # [batch_size, 10]
                                  hand_pose=output[:, 3: 3+45],
                                  global_orient=output[:, 0: 3],
                                  transl=transl)  # [batch_size, 3]

    smplx_joint_pred = mano_pred.joints[:, config.proj_joint_list, :]    # [n, 16, 3]
    joint_img = cam2pixel(smplx_joint_pred, cam[:, 0: 2], cam[:, 2: 4])[:, :, :2]  # [n, 16, 2]

    joint_img = joint_img - joint_img[:,0,:].view(batch_size, 1, 2)    # 和标签中一致，相对于根节点的投影损失

    loss_2D = config.common_loss(joint_img.view(batch_size, -1), kp2d)
    return loss_2D


def Loss_vposer(output, body_pose):
    pred = config.vp_model.encode(output[:, 3: 3+63]).mean
    pred = config.vp_model.decode(pred)['pose_body']

    label = config.vp_model.encode(body_pose).mean
    label = config.vp_model.decode(label)['pose_body']

    # return config.common_loss(pred, label)
    return Loss_rotation_matrix(pred, label)

def batch_rodrigues(rot_vecs, epsilon=1e-8):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device
    dtype = rot_vecs.dtype

    angle = torch.norm(rot_vecs + epsilon, dim=1, keepdim=True, p=2)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

# '''
#     转成实际角度去比较
# '''
# def Loss_rotation_matrix(pred, label):
#     pred_mat = batch_rodrigues(pred.view(-1, 3)).view(-1, 3, 3)
#     label_mat = batch_rodrigues(label.view(-1, 3)).view(-1, 3, 3)
#
#     I = torch.eye(3).view(-1,9)
#     I = I.to(config.device)
#     # loss = torch.nn.SmoothL1Loss()(torch.matmul(pred_mat, torch.transpose(label_mat, 1, 2)), I)    # 这里无需与I矩阵比，直接算乘法后的主对角线元素之和即可
#
#     # 第二种做法
#     delta = torch.matmul(pred_mat, torch.transpose(label_mat, 1, 2)).view(-1, 9)    # 与标签乘法，越接近于I越最优
#     # loss = (torch.sum(delta * I)/repeat_dim/3 - 1)**2   # batch_size个数
#
#     # loss = torch.arccos((torch.sum(delta * I, dim=-1)-1)/2)
#
#     tmp = (torch.sum(delta * I, dim=-1) - 1) / 2
#     tmp = tmp.detach().cpu().numpy().tolist()
#
#     for i in range(len(tmp)):
#         t = tmp[i]
#         if t < -1 or t > 1:
#             print("t:", t)
#             print("delta:", delta[i])
#
#
#     theta = torch.arccos((torch.sum(delta * I, dim=-1) - 1) / 2)
#
#     return config.common_loss(theta, torch.zeros_like(theta))


def Loss_rotation_matrix(pred, label):
    pred_mat = batch_rodrigues(pred.view(-1, 3)).view(-1, 3, 3)
    label_mat = batch_rodrigues(label.view(-1, 3)).view(-1, 3, 3)

    repeat_dim = pred_mat.shape[0]
    I = torch.eye(3).unsqueeze(0).repeat(repeat_dim, 1, 1)
    I = I.to(config.device)
    # loss = torch.nn.SmoothL1Loss()(torch.matmul(pred_mat, torch.transpose(label_mat, 1, 2)), I)    # 这里无需与I矩阵比，直接算乘法后的主对角线元素之和即可

    # 第二种做法
    delta = torch.matmul(pred_mat, torch.transpose(label_mat, 1, 2)).view(-1, 9)
    loss = (torch.sum(delta * I.view(-1, 9))/repeat_dim/3 - 1)**2   # batch_size个数

    return loss

def compute_cov():
    data = np.load("./data/InterHand2.6M_label_hand_pose_train.npy")
    print(data.shape)
    data = data.reshape(data.shape[0], -1)

    # data = np.array([[1, 2, 3],
    #                  [3, 1, 1]], dtype=np.float32)

    # 1、先计算各列的mean
    mean_list = np.mean(data, axis=0)  # axis=0，按列计算

    # 2、原矩阵-meman
    data = data - mean_list

    # 3、计算协方差矩阵
    cov = (data.T @ data) / (data.shape[0] - 1)
    # print(cov)

    #############################################
    # 直接调包计算
    print("######################################")
    npcov = np.cov(data.T)
    # print(npcov)

    result = np.all([cov, npcov])
    print(str(result))

    if str(result) == 'True':
        print("数据计算正确")
        np.save("./interhand_prior.npy", cov)

        eig_value, eig_vector = np.linalg.eig(cov)    # 计算特征值和特征向量
        print("eig_value:", eig_value)
        argSort = np.argsort(eig_value)

        # # 遍历，获取最合适的降维个数
        # for downsample_dim in range(1, 45):
        #     down_args = argSort[:-downsample_dim:-1]
        #     downsample_opt = eig_value[down_args]    # 特征值
        #     print(downsample_dim, sum(downsample_opt), sum(downsample_opt)/sum(eig_value))

        # # 降维
        # downsample_dim = 10    # 99.9%的特征都覆盖到
        # down_argsort = argSort[:-downsample_dim-1:-1]
        # print("down_argsort:", down_argsort)
        #
        # downsample_opt = eig_vector[down_argsort]    # [48, 63]
        #
        # np.save("./h36m_prior_downsample.npy", downsample_opt)


if __name__ == '__main__':
    # batch_size = 1024
    # pred = torch.randn([batch_size, 66], dtype=torch.float32)
    # label = torch.randn([batch_size, 66], dtype=torch.float32)
    #
    # rotation_loss = Loss_rotation_matrix(pred, label)
    # print(rotation_loss)
    compute_cov()



