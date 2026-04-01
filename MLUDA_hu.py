# -*- coding:utf-8 -*-
# Author：Mingshuo Cai
# Create_time：2023-08-01
# Updata_time：2024-03-15
# Usage：Implementation of the MLUDA method on the Houston cross-domain dataset

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import mmd
import numpy as np
from sklearn import metrics
from net2 import DSANSS, DS_MAE
import time
import utils
from torch.utils.data import TensorDataset, DataLoader
from contrastive_loss import SupConLoss
from config_Houston import *
from sklearn import svm
from UtilsCMS import *
from EDL_GOT import edl_loss, ppot_loss


##################################
data_path_s = './datasets/Houston/Houston13.mat'
label_path_s = './datasets/Houston/Houston13_7gt.mat'
data_path_t = './datasets/Houston/Houston18.mat'
label_path_t = './datasets/Houston/Houston18_7gt.mat'

data_s,label_s = utils.load_data_houston(data_path_s,label_path_s)
data_t,label_t = utils.load_data_houston(data_path_t,label_path_t)

TCLDM_Adaptation(data_s, data_t, pca_n, radius)

# Loss Function

DSH_loss = utils.Domain_Occ_loss().cuda()

acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, CLASS_NUM])
k = np.zeros([nDataSet, 1])
best_predict_all = []
best_acc_all = 0.0
best_G,best_RandPerm,best_Row, best_Column,best_nTrain = None,None,None,None,None

for iDataSet in range(nDataSet):
    print('#######################idataset######################## ', iDataSet)
    utils.set_seed(seeds[iDataSet])

    trainX, trainY = utils.get_sample_data(data_s, label_s, HalfWidth, 180)
    testID, testX, testY, G, RandPerm, Row, Column = utils.get_all_data(data_t, label_t, HalfWidth)

    train_dataset = TensorDataset(torch.tensor(trainX), torch.tensor(trainY))
    test_dataset = TensorDataset(torch.tensor(testX), torch.tensor(testY))

    train_loader_s = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    train_loader_t = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
    test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False,drop_last=True)

    len_source_loader = len(train_loader_s)
    len_target_loader = len(train_loader_t)

    # model
    feature_encoder = DSANSS(nBand, patch_size, CLASS_NUM).cuda()

    # =========================================================================
    # 第一阶段：DS-MAE 无监督预训练 (Unsupervised Pre-training Phase)
    # 目标：通过掩蔽重建任务，使网络在无标签状态下内化地物的物理光谱成键规律
    # =========================================================================
    print("=========================================")
    print("Phase 1: Starting DS-MAE Unsupervised Pre-training...")
    # 实例化 MAE，in_channels 接收 Houston 降维后的 nBand
    mae_model = DS_MAE(encoder=feature_encoder, in_channels=nBand, embed_dim=288, mask_ratio=0.75).cuda()
    mae_optimizer = torch.optim.Adam(mae_model.parameters(), lr=1e-3, weight_decay=1e-5)

    pretrain_epochs = 20  # 预训练轮次，可根据数据集大小灵活调整 (10~30轮)
    num_iter = len_source_loader  # 提前定义全局迭代次数，消除作用域未定义错误

    for p_epoch in range(pretrain_epochs):
        mae_model.train()
        total_mae_loss = 0.0

        # 修正：使用正确的数据加载器名称 train_loader_s 与 train_loader_t
        source_train_iter_mae = iter(train_loader_s)
        target_train_iter_mae = iter(train_loader_t)

        for i in range(1, num_iter):
            # 获取源域数据
            try:
                source_data, _ = next(source_train_iter_mae)
            except StopIteration:
                source_train_iter_mae = iter(train_loader_s)
                source_data, _ = next(source_train_iter_mae)

            # 获取目标域数据
            try:
                target_data, _ = next(target_train_iter_mae)
            except StopIteration:
                target_train_iter_mae = iter(train_loader_t)
                target_data, _ = next(target_train_iter_mae)

            mae_optimizer.zero_grad()
            # 执行高比例空间与光谱联合掩蔽及重建
            loss_mae = mae_model(source_data.cuda(), target_data.cuda())
            loss_mae.backward()
            mae_optimizer.step()
            total_mae_loss += loss_mae.item()

        print('DS-MAE Pre-train Epoch {:>3d}:   MSE Loss: {:6.4f}'.format(p_epoch + 1, total_mae_loss / num_iter))

    print("Phase 1 Completed: Feature Extractor initialized with deep physical semantics.")
    print("=========================================\n")

    # =========================================================================
    # 第二阶段：主 UDA 微调流水线 (Main UDA Fine-tuning Pipeline)
    # =========================================================================
    print("Phase 2: Training UDA Pipeline...")

    last_accuracy = 0.0
    best_episdoe = 0
    train_loss = []
    test_acc = []
    running_D_loss, running_F_loss = 0.0, 0.0
    running_label_loss = 0
    running_domain_loss = 0
    total_hit, total_num = 0.0, 0.0
    size = 0.0
    test_acc_list = []

    train_start = time.time()

    #loss plot
    loss1 = []
    loss2 = []
    loss3 = []

    for epoch in range(1, epochs + 1):
        LEARNING_RATE = lr / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)
        print('learning rate{: .4f}'.format(LEARNING_RATE))
        optimizer = torch.optim.SGD([
            {'params': feature_encoder.feature_layers.parameters(), },
            {'params': feature_encoder.fc1.parameters(), 'lr': LEARNING_RATE},
            {'params': feature_encoder.fc2.parameters(), 'lr': LEARNING_RATE},
            {'params': feature_encoder.head1.parameters(), 'lr': LEARNING_RATE},
            {'params': feature_encoder.head2.parameters(), 'lr': LEARNING_RATE},
            # 【致命错误修复 4】：必须将因果模块注册进优化器！
            {'params': feature_encoder.causal_disentangle.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE, momentum=momentum, weight_decay=l2_decay)

        feature_encoder.train()

        iter_source = iter(train_loader_s)
        iter_target = iter(train_loader_t)
        num_iter = len_source_loader

        for i in range(1,num_iter):
            source_data, source_label = next(iter_source)
            target_data, target_label = next(iter_target)

            if i % len_target_loader == 0:
                iter_target = iter(train_loader_t)

            # 0
            source_data0 = utils.radiation_noise(source_data)
            source_data0 = source_data0.type(torch.FloatTensor)
            # 1
            source_data1 = utils.flip_augmentation(source_data)
            # 2
            target_data0 = utils.radiation_noise(target_data)
            target_data0 = target_data0.type(torch.FloatTensor)
            # 3
            target_data1 = utils.flip_augmentation(target_data)

            (source_features, source1, _, source_outputs, source_out,
             target_features, _, target1, target_outputs, target_out, causal_loss) = feature_encoder(source_data.cuda(),
                                                                                        target_data.cuda())
            # 删除了 source2, source3, target2, target3 的冗余推理，消除SCL显存开销

            # 动态权重调节
            lambd = 2 / (1 + math.exp(-10 * (epoch) / epochs)) - 1

            # 1. 逻辑革新：证据深度学习分类损失 (替代交叉熵)
            cls_loss = edl_loss(source_outputs, source_label.cuda(), epoch, epochs, CLASS_NUM)

            # 2. 对齐革新：概率极化的几何最优传输 (替代LMMD与SCL，包含负伪标签校验)
            got_loss = ppot_loss(source_features, target_features, source_label.cuda(), target_outputs, epoch, epochs,
                                 CLASS_NUM)

            # 最终损失融合
            loss = cls_loss + lambd * got_loss + 0.1 * causal_loss
            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = source_outputs.data.max(1)[1]
            total_hit += pred.eq(source_label.data.cuda()).sum()
            size += source_label.data.size()[0]

            test_accuracy = 100. * float(total_hit) / size

        print(
            'epoch {:>3d}:   cls loss: {:6.4f}, got loss: {:6.4f}, causal loss: {:6.4f}, acc {:6.4f}, total loss: {:6.4f}'
            .format(epoch, cls_loss.item(), got_loss.item(), causal_loss.item(), total_hit / size, loss.item()))


        train_end = time.time()
        if epoch % epochs == 0:
            # print("Testing ...")
            feature_encoder.eval()
            total_rewards = 0
            counter = 0
            accuracies = []
            predict = np.array([], dtype=np.int64)
            labels = np.array([], dtype=np.int64)
            with torch.no_grad():
                for test_datas, test_labels in test_loader:
                    batch_size = test_labels.shape[0]

                    source_features, source1, _, source_outputs, source_out, test_features, _, _, test_outputs, _, _ = feature_encoder(
                            Variable(source_data).cuda(), Variable(test_datas).cuda())

                    pred = test_outputs.data.max(1)[1]

                    test_labels = test_labels.numpy()
                    rewards = [1 if pred[j] == test_labels[j] else 0 for j in range(batch_size)]

                    total_rewards += np.sum(rewards)
                    counter += batch_size

                    predict = np.append(predict, pred.cpu().numpy())
                    labels = np.append(labels, test_labels)

                    accuracy = total_rewards / 1.0 / counter  #
                    accuracies.append(accuracy)

            test_accuracy = 100. * total_rewards / len(test_loader.dataset)
            acc[iDataSet] = 100. * total_rewards / len(test_loader.dataset)
            OA = acc
            C = metrics.confusion_matrix(labels, predict)
            A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float64)

            k[iDataSet] = metrics.cohen_kappa_score(labels, predict)
            print('\t\tAccuracy: {}/{} ({:.2f}%)\n'.format(total_rewards, len(test_loader.dataset),
                                                           100. * total_rewards / len(test_loader.dataset)))
            test_end = time.time()

            # Training mode

            if test_accuracy > last_accuracy:
                # save networks
                # torch.save(feature_encoder.state_dict(),str("../checkpoints/DFSL_feature_encoder_" + "houston_cl_lmmd_dis_attention" +str(iDataSet) +".pkl"))
                print("save networks for epoch:", epoch + 1)
                last_accuracy = test_accuracy
                best_episdoe = epoch
                best_predict_all = predict
                best_G, best_RandPerm, best_Row, best_Column = G, RandPerm, Row, Column
                print('best epoch:[{}], best accuracy={}'.format(best_episdoe + 1, last_accuracy))

            print('iter:{} best epoch:[{}], best accuracy={}'.format(iDataSet, best_episdoe + 1, last_accuracy))
            print('***********************************************************************************')

AA = np.mean(A, 1)
AAMean = np.mean(AA,0)
AAStd = np.std(AA)
AMean = np.mean(A, 0)
AStd = np.std(A, 0)
OAMean = np.mean(acc)
OAStd = np.std(acc)
kMean = np.mean(k)
kStd = np.std(k)
print ("train time per DataSet(s): " + "{:.5f}".format(train_end-train_start))
print("test time per DataSet(s): " + "{:.5f}".format(test_end-train_end))
print ("average OA: " + "{:.2f}".format( OAMean) + " +- " + "{:.2f}".format( OAStd))
print ("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
print ("average kappa: " + "{:.4f}".format(100 *kMean) + " +- " + "{:.4f}".format(100 *kStd))
print ("accuracy for each class: ")
for i in range(CLASS_NUM):
    print ("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))

best_iDataset = 0
for i in range(len(acc)):
    print('{}:{}'.format(i, acc[i]))
    if acc[i] > acc[best_iDataset]:
        best_iDataset = i
print('best acc all={}'.format(acc[best_iDataset]))

#################classification map################################

for i in range(len(best_predict_all)):  # predict ndarray <class 'tuple'>: (9729,)
    best_G[best_Row[best_RandPerm[ i]]][best_Column[best_RandPerm[ i]]] = best_predict_all[i] + 1

hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))
for i in range(best_G.shape[0]):
    for j in range(best_G.shape[1]):
        if best_G[i][j] == 0:
            hsi_pic[i, j, :] = [0, 0, 0]
        if best_G[i][j] == 1:
            hsi_pic[i, j, :] = [0, 0, 1]
        if best_G[i][j] == 2:
            hsi_pic[i, j, :] = [0, 1, 0]
        if best_G[i][j] == 3:
            hsi_pic[i, j, :] = [0, 1, 1]
        if best_G[i][j] == 4:
            hsi_pic[i, j, :] = [1, 0, 0]
        if best_G[i][j] == 5:
            hsi_pic[i, j, :] = [1, 0, 1]
        if best_G[i][j] == 6:
            hsi_pic[i, j, :] = [1, 1, 0]
        if best_G[i][j] == 7:
            hsi_pic[i, j, :] = [0.5, 0.5, 1]

utils.classification_map(hsi_pic[4:-4, 4:-4, :], best_G[4:-4, 4:-4], 24,  "classificationMap/housotn18.png")
