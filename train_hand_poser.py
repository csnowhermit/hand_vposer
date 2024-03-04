from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time

from dataset import HandData
from hand_poser import HandPoser
from common_utils import Loss_rotation_matrix

'''
    损失函数：
    1.重建损失：
    2.正则化损失：避免过拟合
'''

os.environ['CUDA_VISIBEL_DEVICES'] = '0'

writer = SummaryWriter()

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Batch size during training
batch_size = 1024
num_epochs = 200
lr = 1e-4
ngpu = 1
alpha_proj2d = 0.01
alpha_prior = 0.01

num_neurons = 512
latentD = 23

root_path = "./data/"
train_dataset = HandData(root_path='./data',
                         label_hand_pose_file_list=['InterHand2.6M_label_hand_pose_train.npy'])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = HandData(root_path='./data',
                       label_hand_pose_file_list=['InterHand2.6M_label_hand_pose_val.npy'])
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# 自动创建保存模型的目录
if os.path.exists(os.path.join(root_path, "checkpoint/")) is False:
    os.makedirs(os.path.join(root_path, "checkpoint/"))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = HandPoser(num_neurons=num_neurons, latentD=latentD)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
scheduler = lr_scheduler.StepLR(optimizer, step_size=len(train_dataloader) * 10, gamma=0.9)  # 每10 epoch衰减到0.9

best_eval_loss = 999999  # 最好的loss，val loss mean只要比该值小，则保存

total_step = len(train_dataloader)
model.train()
for epoch in range(num_epochs):
    start = time.time()

    for i, (hand_pose) in enumerate(train_dataloader):
        optimizer.zero_grad()
        hand_pose = hand_pose.to(device)

        latent_distribution = model.encode(hand_pose)
        latent_locs = latent_distribution.mean
        pred = model.decode(latent_locs)

        latent_smapling = latent_distribution.rsample()
        pred_bias = model.decode(latent_smapling)

        ### Principal Loss
        loss_recon = Loss_rotation_matrix(pred, hand_pose)
        loss_recon_bias = 0.5 * Loss_rotation_matrix(pred_bias, hand_pose)
        loss_recon_similarity = 0.5 * Loss_rotation_matrix(pred_bias, pred)

        ### Distribution Loss
        latent_locs_mean = latent_locs.mean(dim=0)  # latent_locs_mean 大球的mean；  latent_locs 多个小球的mean
        latent_locs_cov = torch.cov(latent_locs.transpose(0, 1))  # 多个小球mean的方差组成大球的方差

        # loss_locs_mean = nn.L1Loss()(latent_locs_mean, torch.zeros(latentD))  # miu 和0比
        # loss_locs_cov = nn.L1Loss()(latent_locs_cov, torch.eye(latentD))    # sigma 和I比
        KL_locs_global = -0.1 * 0.5 * (
                    torch.log(torch.linalg.det(latent_locs_cov)) - torch.trace(latent_locs_cov) - torch.norm(
                latent_locs_mean) + latentD)  # 让大球接近于N(0, Identity)

        setting_cov = 1 / 4 * torch.exp(-torch.norm(latent_locs, dim=1) / 2)

        local_covariance = latent_distribution.variance

        KL_local_ditribution = -0.1 * 0.5 * (
                    torch.sum(torch.log(local_covariance), dim=-1) - latentD * torch.log(setting_cov) - torch.sum(
                local_covariance, dim=-1) + latentD).mean()  # 越远离大球中心，小球的半径应该越小

        loss = loss_recon + loss_recon_bias + loss_recon_similarity + KL_locs_global + KL_local_ditribution

        loss.backward()
        optimizer.step()
        scheduler.step()

        curr_step = epoch * total_step + i
        writer.add_scalar("train/loss_recon", loss_recon, curr_step)
        writer.add_scalar("train/loss_recon_bias", loss_recon_bias, curr_step)
        writer.add_scalar("train/loss_recon_similarity", loss_recon_similarity, curr_step)
        writer.add_scalar("train/KL_locs", KL_locs_global, curr_step)
        writer.add_scalar("train/KL_local_ditribution", KL_local_ditribution, curr_step)
        writer.flush()

        if (i % 10) == 0:
            print(
                'Epoch [{}/{}], Step [{}/{}], loss_recon: {:.4f}, loss_recon_bias: {:.4f}, loss_recon_similarity: {:.4f}, KL_locs: {:.4f}, KL_local_ditribution: {:.4f}, spend time: {:.4f}'
                .format(epoch + 1, num_epochs, i + 1, total_step, loss_recon.item(), loss_recon_bias.item(),
                        loss_recon_similarity.item(), KL_locs_global.item(), KL_local_ditribution.item(),
                        time.time() - start))
            start = time.time()

    model.eval()
    with torch.no_grad():
        val_loss_recon = []
        val_loss_recon_bias = []
        val_loss_recon_similarity = []
        val_KL_locs = []
        val_KL_local_ditribution = []

        start = time.time()
        for i, (hand_pose) in enumerate(val_dataloader):
            optimizer.zero_grad()
            hand_pose = hand_pose.to(device)

            latent_distribution = model.encode(hand_pose)
            latent_locs = latent_distribution.mean
            pred = model.decode(latent_locs)

            latent_smapling = latent_distribution.rsample()
            pred_bias = model.decode(latent_smapling)

            ### Principal Loss
            loss_recon = Loss_rotation_matrix(pred, hand_pose)
            loss_recon_bias = 0.5 * Loss_rotation_matrix(pred_bias, hand_pose)
            loss_recon_similarity = 0.5 * Loss_rotation_matrix(pred_bias, pred)

            ### Distribution Loss
            latent_locs_mean = latent_locs.mean(dim=0)  # latent_locs_mean 大球的mean；  latent_locs 多个小球的mean
            latent_locs_cov = torch.cov(latent_locs.transpose(0, 1))  # 多个小球mean的方差组成大球的方差

            # loss_locs_mean = nn.L1Loss()(latent_locs_mean, torch.zeros(latentD))  # miu 和0比
            # loss_locs_cov = nn.L1Loss()(latent_locs_cov, torch.eye(latentD))    # sigma 和I比
            KL_locs_global = -0.1 * 0.5 * (
                        torch.log(torch.linalg.det(latent_locs_cov)) - torch.trace(latent_locs_cov) - torch.norm(
                    latent_locs_mean) + latentD)  # 让大球接近于N(0, Identity)

            setting_cov = 1 / 4 * torch.exp(-torch.norm(latent_locs, dim=1) / 2)

            local_covariance = latent_distribution.variance

            KL_local_ditribution = -0.1 * 0.5 * (
                        torch.sum(torch.log(local_covariance), dim=-1) - latentD * torch.log(setting_cov) - torch.sum(
                    local_covariance, dim=-1) + latentD).mean()  # 越远离大球中心，小球的半径应该越小

            ### Total Loss

            loss = loss_recon + loss_recon_bias + loss_recon_similarity + KL_locs_global + KL_local_ditribution

            val_loss_recon.append(loss_recon.item())
            val_loss_recon_bias.append(loss_recon_bias.item())
            val_loss_recon_similarity.append(loss_recon_similarity.item())
            val_KL_locs.append(KL_locs_global.item())
            val_KL_local_ditribution.append(KL_local_ditribution.item())

        curr_recon_loss = np.mean(val_loss_recon)
        curr_recon_bias_loss = np.mean(val_loss_recon_bias)
        curr_recon_similarity_loss = np.mean(val_loss_recon_similarity)
        curr_KL_locs_loss = np.mean(val_KL_locs)
        curr_KL_local_ditribution_loss = np.mean(val_KL_local_ditribution)

        print(
            'Epoch [{}/{}], loss_recon: {:.4f}, loss_recon_bias: {:.4f}, loss_recon_similarity: {:.4f}, KL_locs: {:.4f}, KL_local_ditribution: {:.4f}, time: {:.4f}'
            .format(epoch + 1, num_epochs, curr_recon_loss, curr_recon_bias_loss, curr_recon_similarity_loss,
                    curr_KL_locs_loss, curr_KL_local_ditribution_loss, time.time() - start))

        writer.add_scalar("val/loss_recon", curr_recon_loss, epoch)
        writer.add_scalar("val/loss_recon_bias", curr_recon_bias_loss, epoch)
        writer.add_scalar("val/loss_recon_similarity", curr_recon_similarity_loss, epoch)
        writer.add_scalar("val/KL_locs", curr_KL_locs_loss, epoch)
        writer.add_scalar("val/KL_local_ditribution", curr_KL_local_ditribution_loss, epoch)
        writer.flush()

        if curr_recon_loss < best_eval_loss or epoch % 10 == 0:  # 只要损失下降就保存
            best_eval_loss = curr_recon_loss  # 保存当前的loss为最好
            torch.save({
                "curr_epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_eval_loss": best_eval_loss,
                "lr": scheduler.get_last_lr()
            }, root_path + 'checkpoint/hand_poser_{}_loss_{:.6f}.pt'.format(epoch, curr_recon_loss))

    model.train()
