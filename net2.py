# -*- coding:utf-8 -*-
# Author：Mingshuo Cai
# Create_time：2023-08-01
# Updata_time：2024-03-15
# Usage：Implementation of the Cross attention proposed in MLUDA.

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class DSANSS(nn.Module):
    def __init__(self, n_band=198, patch_size=3,num_class=3):
        super(DSANSS, self).__init__()
        self.n_outputs = 288
        self.feature_layers = DCRN_02(n_band,patch_size,num_class)

        # 实例化因果特征解耦模块
        self.causal_disentangle = CausalDisentanglement(dim=288)

        self.fc1 = nn.Linear(288, num_class)
        self.fc2 = nn.Linear(288, 1)

        self.head1 = nn.Sequential(
            nn.Linear(288, 128),
            # nn.ReLU(inplace=True),
            # nn.Linear(288, 128)
        )
        self.head2 = nn.Sequential(
            nn.Linear(288, 128),
            # nn.ReLU(inplace=True),
            # nn.Linear(288, 128)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        features_x, features_y = self.feature_layers(x, y)

        # 应用因果屏蔽：隔离虚假相关性，提取纯粹的因果特征
        causal_feat_x, indep_loss_x = self.causal_disentangle(features_x)
        causal_feat_y, indep_loss_y = self.causal_disentangle(features_y)

        causal_loss = indep_loss_x + indep_loss_y

        # 下游模块（包括 fc1 的 PPOT 对齐输入与 fc2 的分类）均仅接收因果特征
        x1 = F.normalize(self.head1(causal_feat_x), dim=1)
        x2 = F.normalize(self.head2(causal_feat_x), dim=1)
        fea_x = self.fc1(causal_feat_x)
        output_x = self.fc2(causal_feat_x)
        output_x = self.sigmoid(output_x)

        y1 = F.normalize(self.head1(causal_feat_y), dim=1)
        y2 = F.normalize(self.head2(causal_feat_y), dim=1)
        fea_y = self.fc1(causal_feat_y)
        output_y = self.fc2(causal_feat_y)
        output_y = self.sigmoid(output_y)

        # 返回值追加 causal_loss，向上传递以更新梯度
        return causal_feat_x, x1, x2, fea_x, output_x, causal_feat_y, y1, y2, fea_y, output_y, causal_loss

    def get_embedding(self, x):
        out, _, _, _, _ = self.forward(x)
        return out

class DSAN1(nn.Module):
    def __init__(self, n_band=198, patch_size=3,num_class=3):
        super(DSAN1, self).__init__()
        self.n_outputs = 288
        self.feature_layers = DCRN_02(n_band,patch_size,num_class)

        self.fc1 = nn.Linear(288, num_class)
        self.fc2 = nn.Linear(288, 1)

        self.head1 = nn.Sequential(
            nn.Linear(288, 64),
            # nn.ReLU(inplace=True),
            # nn.Linear(288, 128)
        )

        self.head2 = nn.Sequential(
            nn.Linear(288, 64),
            # nn.ReLU(inplace=True),
            # nn.Linear(288, 128)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        features = self.feature_layers(x)

        x1 = F.normalize(self.head1(features), dim=1)
        x2 = F.normalize(self.head2(features), dim=1)

        fea = self.fc1(features)
        output = self.fc2(features)
        output = self.sigmoid(output)

        return features,x1,x2,fea, output

    def get_embedding(self, x):
        out, _, _, _, _ = self.forward(x)
        return out

class DSAN2(nn.Module):
    def __init__(self, n_band=198, patch_size=3,num_class=3):
        super(DSAN1, self).__init__()
        self.n_outputs = 152
        self.feature_layers = DCRN(n_band,patch_size,num_class)

        self.fc1 = nn.Linear(self.n_outputs, num_class)
        self.fc2 = nn.Linear(self.n_outputs, 1)

        self.head1 = nn.Sequential(
            nn.Linear(self.n_outputs, 128),
            # nn.ReLU(inplace=True),
            # nn.Linear(288, 128)
        )
        self.head2 = nn.Sequential(
            nn.Linear(self.n_outputs, 128),
            # nn.ReLU(inplace=True),
            # nn.Linear(288, 128)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        features = self.feature_layers(x)

        x1 = F.normalize(self.head1(features), dim=1)
        x2 = F.normalize(self.head2(features), dim=1)

        fea = self.fc1(features)
        output = self.fc2(features)
        output = self.sigmoid(output)

        return features,x1,x2,fea, output

    def get_embedding(self, x):
        out, _, _, _, _ = self.forward(x)
        return out

class DCRN_02(nn.Module):
    # CMS used
    def __init__(self, input_channels, patch_size, n_classes):
        super(DCRN_02, self).__init__()
        self.kernel_dim = 1
        self.feature_dim = input_channels
        self.sz = patch_size
        # Convolution Layer 1 kernel_size = (1, 1, 7), stride = (1, 1, 2), output channels = 24
        self.conv1 = nn.Conv3d(1, 24, kernel_size=(7, 1, 1), stride=(2, 1, 1), bias=True)
        self.bn1 = nn.BatchNorm3d(24)
        self.activation1 = nn.ReLU()

        # Residual block 1
        self.conv2 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0),
                               bias=True)  # padding_mode='replicate',
        self.bn2 = nn.BatchNorm3d(24)
        self.activation2 = nn.ReLU()

        self.conv3 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0),
                               bias=True)  # padding_mode='replicate',
        self.bn3 = nn.BatchNorm3d(24)
        self.activation3 = nn.ReLU()
        # Finish

        # Convolution Layer 2 kernel_size = (1, 1, (self.feature_dim - 6) // 2), output channels = 128
        self.conv4 = nn.Conv3d(24, 192, kernel_size=(((self.feature_dim - 7) // 2 + 1), 1, 1), bias=True)
        self.bn4 = nn.BatchNorm3d(192)
        self.activation4 = nn.ReLU()

        # Convolution layer for spatial information
        self.conv5 = nn.Conv3d(1, 24, (self.feature_dim, 1, 1))
        self.bn5 = nn.BatchNorm3d(24)
        self.activation5 = nn.ReLU()

        # Residual block 2
        self.conv6 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                               bias=True)  # padding_mode='replicate',
        self.bn6 = nn.BatchNorm3d(24)
        self.activation6 = nn.ReLU()

        self.conv7 = nn.Conv3d(24, 96, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                               bias=True)  # padding_mode='replicate',
        self.bn7 = nn.BatchNorm3d(96)
        self.activation7 = nn.ReLU()

        self.conv8 = nn.Conv3d(24, 96, kernel_size=1)

        # Finish

        # Combination shape
        # self.inter_size = 128 + 24
        self.inter_size = 192 + 96


        # Residual block 3
        self.conv9 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                               bias=True)  # padding_mode='replicate',
        self.bn9 = nn.BatchNorm3d(self.inter_size)
        self.activation9 = nn.ReLU()
        self.conv10 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                                bias=True)  # padding_mode='replicate',
        self.bn10 = nn.BatchNorm3d(self.inter_size)
        self.activation10 = nn.ReLU()

        # attention
        self.ca = ChannelAttention(self.inter_size)
        self.sa = SpatialAttention()

        # Average pooling kernel_size = (5, 5, 1)
        self.avgpool = nn.AvgPool3d((1, self.sz, self.sz))

        # Fully connected Layer
        self.fc1 = nn.Linear(in_features=self.inter_size, out_features=n_classes)

        self.cd_bimamba = CD_BiMamba(dim=self.inter_size, r=8)
        # parameters initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x,y):
        x = x.unsqueeze(1)  # (64,1,100,9,9)
        x1 = self.conv1(x)
        x1 = self.activation1(self.bn1(x1))   
        residual = x1
        x1 = self.conv2(x1)
        x1 = self.activation2(self.bn2(x1))
        x1 = self.conv3(x1)
        x1 = residual + x1  # (32,24,21,7,7)
        x1 = self.activation3(self.bn3(x1))
        
        
        # Convolution layer to combine rest
        x1 = self.conv4(x1)  # (32,128,1,7,7)
        x1 = self.activation4(self.bn4(x1))
        x1 = x1.reshape(x1.size(0), x1.size(1), x1.size(3), x1.size(4))  # (32,128,7,7)
        
        x2 = self.conv5(x)  # (32,24,1,7,7)
        x2 = self.activation5(self.bn5(x2))
        # Residual layer 2
        residual = x2
        residual = self.conv8(residual)  # (32,24,1,7,7)
        x2 = self.conv6(x2)  # (32,24,1,7,7)
        x2 = self.activation6(self.bn6(x2))
        x2 = self.conv7(x2)  # (32,24,1,7,7)
        x2 = residual + x2
        x2 = self.activation7(self.bn7(x2))
        x2 = x2.reshape(x2.size(0), x2.size(1), x2.size(3), x2.size(4))  # (32,24,7,7)
        
        y = y.unsqueeze(1)  # (64,1,100,9,9)
        # Convolution layer 1
        y1 = self.conv1(y)
        y1 = self.activation1(self.bn1(y1))   # 直接activation+Relu
        # Residual layer 1
        residual = y1
        y1 = self.conv2(y1)
        y1 = self.activation2(self.bn2(y1))
        y1 = self.conv3(y1)
        y1 = residual + y1  # (32,24,21,7,7)
        y1 = self.activation3(self.bn3(y1))

        # Convolution layer to combine rest
        y1 = self.conv4(y1)  # (32,128,1,7,7)
        y1 = self.activation4(self.bn4(y1))
        y1 = y1.reshape(y1.size(0), y1.size(1), y1.size(3), y1.size(4))  # (32,128,7,7)
        y2 = self.conv5(y)  # (32,24,1,7,7)
        y2 = self.activation5(self.bn5(y2))
        # Residual layer 2
        residual = y2
        residual = self.conv8(residual)  # (32,24,1,7,7)
        y2 = self.conv6(y2)  # (32,24,1,7,7)
        y2 = self.activation6(self.bn6(y2))
        y2 = self.conv7(y2)  # (32,24,1,7,7)
        y2 = residual + y2
        y2 = self.activation7(self.bn7(y2))
        y2 = y2.reshape(y2.size(0), y2.size(1), y2.size(3), y2.size(4))  # (32,24,7,7)

        x = torch.cat((x1, x2), 1)  # (32,152,7,7)
        ca_x = self.ca(x)
        sa_x = self.sa(x)

        y = torch.cat((y1, y2), 1)  # (32,152,7,7)
        ca_y = self.ca(y)
        sa_y = self.sa(y)
        lamd = 0.9
        x = ca_x * x
        x = sa_x * x
        y = ca_y * y
        y = sa_y * y

        # 在池化降维前，输入全分辨率三维特征块到 CD-BiMamba 提取全局上下文
        F_x_map, F_y_map = self.cd_bimamba(x, y)

        # 域自适应特征交互完成后，执行池化满足后续全连接层的降维需求
        F_y2x = self.avgpool(F_x_map)
        F_y2x = F_y2x.view(F_y2x.shape[0], -1)  # 对应源特征输出表示

        F_x2y = self.avgpool(F_y_map)
        F_x2y = F_x2y.view(F_x2y.shape[0], -1)  # 对应目标特征输出表示

        return F_y2x, F_x2y


class LoRALinear(nn.Module):
    """参数高效微调（PEFT）：低秩自适应层"""

    def __init__(self, in_features, out_features, r=8, lora_alpha=16):
        super(LoRALinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        # 冻结预训练的主干参数
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        self.r = r
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            self.scaling = lora_alpha / r
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = self.linear(x)
        if self.r > 0:
            lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
            return base_out + lora_out
        return base_out


class SASF_Module(nn.Module):
    """结构感知状态融合机制 (Structure-Aware State Fusion)"""

    def __init__(self, dim, r=8):
        super(SASF_Module, self).__init__()
        # 共享的可学习跨域状态转移矩阵 A_shared
        self.A_shared = nn.Parameter(torch.randn(dim, dim) / math.sqrt(dim))

        # 门控调制网络 (使用 PEFT)
        self.gate = nn.Sequential(
            LoRALinear(dim * 2, dim, r=r),
            nn.Sigmoid()
        )

    def forward(self, h_target, h_source):
        # 目标域隐状态基于源域隐状态语义线索进行动态校准
        gate_val = self.gate(torch.cat([h_target, h_source], dim=-1))
        # 状态流形转移
        transferred_state = h_source @ self.A_shared.T
        # 融合与更新
        h_target_fused = gate_val * h_target + transferred_state
        return h_target_fused


class CD_BiMamba(nn.Module):
    """跨域双向结构感知Mamba网络核心层"""

    def __init__(self, dim, r=8):
        super(CD_BiMamba, self).__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # 独立特征投影层 (集成 LoRA/KronA+)
        self.proj_x = LoRALinear(dim, dim, r=r)
        self.proj_y = LoRALinear(dim, dim, r=r)

        # 双向 SASF 状态融合
        self.sasf_y2x = SASF_Module(dim, r=r)
        self.sasf_x2y = SASF_Module(dim, r=r)

        # 融合后输出投影层
        self.out_proj_x = LoRALinear(dim, dim, r=r)
        self.out_proj_y = LoRALinear(dim, dim, r=r)

    def forward(self, x_map, y_map):
        B, C, H, W = x_map.shape

        # 1. 多向空间-光谱选择性扫描 (Cross-Scan): 将三维张量展开为一维序列
        # 此处以最基本的空间展平代表扫描方向，实际应用可拓展多级重排
        x_seq = x_map.view(B, C, -1).transpose(1, 2)  # [B, L, C]
        y_seq = y_map.view(B, C, -1).transpose(1, 2)  # [B, L, C]

        x_norm = self.norm(x_seq)
        y_norm = self.norm(y_seq)

        # Mamba 隐状态映射
        h_x = self.proj_x(x_norm)
        h_y = self.proj_y(y_norm)

        # 2. 结构感知状态融合 (线性复杂度计算)
        # 源域状态向目标域传递
        h_y_fused = self.sasf_x2y(h_target=h_y, h_source=h_x)
        # 目标域状态反传至源域（双向机制）
        h_x_fused = self.sasf_y2x(h_target=h_x, h_source=h_y)

        # 输出映射并施加残差连接
        out_x = self.out_proj_x(h_x_fused) + x_seq
        out_y = self.out_proj_y(h_y_fused) + y_seq

        # 3. 逆向扫描恢复空间结构
        out_x_map = out_x.transpose(1, 2).view(B, C, H, W)
        out_y_map = out_y.transpose(1, 2).view(B, C, H, W)

        return out_x_map, out_y_map
class DCRN(nn.Module):

    def __init__(self, input_channels, patch_size, n_classes):
        super(DCRN, self).__init__()
        self.kernel_dim = 1                 
        self.feature_dim = input_channels  
        self.sz = patch_size                

        self.conv1 = nn.Conv3d(1, 24, kernel_size=(7, 1, 1), stride=(2, 1, 1), bias=True)
        self.bn1 = nn.BatchNorm3d(24)
        self.activation1 = nn.ReLU()

        self.conv2 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0), bias=True)#padding_mode='replicate',
        self.bn2 = nn.BatchNorm3d(24)
        self.activation2 = nn.ReLU()
        self.conv3 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0),bias=True)# padding_mode='replicate',
        self.bn3 = nn.BatchNorm3d(24)
        self.activation3 = nn.ReLU()
        # Finish
        self.conv4 = nn.Conv3d(24, 128, kernel_size=(((self.feature_dim - 7) // 2 + 1), 1, 1), bias=True)
        self.bn4 = nn.BatchNorm3d(128)
        self.activation4 = nn.ReLU()

        # Convolution layer for spatial information
        self.conv5 = nn.Conv3d(1, 24, (self.feature_dim, 1, 1))
        self.bn5 = nn.BatchNorm3d(24)
        self.activation5 = nn.ReLU()

        # Residual block 2
        self.conv6 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True)#padding_mode='replicate',
        self.bn6 = nn.BatchNorm3d(24)
        self.activation6 = nn.ReLU()
        self.conv7 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True)#padding_mode='replicate',
        self.bn7 = nn.BatchNorm3d(24)
        self.activation7 = nn.ReLU()
        self.conv8 = nn.Conv3d(24, 24, kernel_size=1)
        
        self.inter_size = 128 + 24

        # Residual block 3
        self.conv9 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True)#padding_mode='replicate',
        self.bn9 = nn.BatchNorm3d(self.inter_size)
        self.activation9 = nn.ReLU()
        self.conv10 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),bias=True)#padding_mode='replicate',
        self.bn10 = nn.BatchNorm3d(self.inter_size)
        self.activation10 = nn.ReLU()

        # attention
        self.ca = ChannelAttention(self.inter_size)#self.inter_size
        self.sa = SpatialAttention()

        # Average pooling kernel_size = (5, 5, 1)
        self.avgpool = nn.AvgPool3d((1, self.sz, self.sz))

        # Fully connected Layer
        self.fc1 = nn.Linear(in_features=self.inter_size, out_features=n_classes)

        # 定义参数的初始化形式
        for m in self.modules():
            if isinstance(m, nn.Conv3d):    
                torch.nn.init.kaiming_normal_(m.weight.data)   
                m.bias.data.zero_()                         
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:                
            nn.init.xavier_uniform_(m.weight, gain=1)   
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight, 1.0, 0.02)
            m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data = torch.ones(m.bias.data.size())



    def forward(self, x,y):

        x = x.unsqueeze(1) # (64,1,100,9,9)  -> (64,100,9,9)
        # Convolution layer 1
        x1 = self.conv1(x)
        x1 = self.activation1(self.bn1(x1))
        # Residual layer 1
        residual = x1                                                                                                                                                                                                                                                                                                                                
        x1 = self.conv2(x1)
        x1 = self.activation2(self.bn2(x1))
        x1 = self.conv3(x1)
        x1 = residual + x1                  #(32,24,21,7,7)
        x1 = self.activation3(self.bn3(x1))

        # Convolution layer to combine rest
        x1 = self.conv4(x1)                 #(32,128,1,7,7)
        x1 = self.activation4(self.bn4(x1))
        x1 = x1.reshape(x1.size(0), x1.size(1), x1.size(3), x1.size(4)) #(32,128,7,7)

        x2 = self.conv5(x)                      #(32,24,1,7,7)
        x2 = self.activation5(self.bn5(x2))

        # Residual layer 2
        residual = x2
        residual = self.conv8(residual)     #(32,24,1,7,7)
        x2 = self.conv6(x2)                 #(32,24,1,7,7)
        x2 = self.activation6(self.bn6(x2))
        x2 = self.conv7(x2)                 #(32,24,1,7,7)
        x2 = residual + x2

        x2 = self.activation7(self.bn7(x2))
        x2 = x2.reshape(x2.size(0), x2.size(1), x2.size(3), x2.size(4)) #(32,24,7,7)

        x = torch.cat((x1, x2), 1)      #(32,152,7,7)


        ###################
        # attention map
        ###################

        x = self.ca(x) * x                  
        x = self.sa(x) * x                  

        x = self.avgpool(x)                 
        x = x.view(x.shape[0], -1) 

        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 4, 1, bias=False) #4-->16
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 4, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CausalDisentanglement(nn.Module):
    """
    基于结构因果模型 (SCM) 的跨域因果特征解耦模块
    核心修复版：梯度隔离，防止 SCM 正则化反噬主干网络
    """

    def __init__(self, dim):
        super(CausalDisentanglement, self).__init__()
        # 因果掩码生成器
        self.mask_generator = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Sigmoid()
        )

    def forward(self, h):
        # 【致密修复】：使用 h.detach()！
        # 强制截断梯度。掩码生成器可以“看”到特征来预测掩码，
        # 但它产生的正则化损失（indep_loss 等）绝对不能反向破坏 h 原本的语义分布。
        causal_mask = self.mask_generator(h.detach())

        # 提取因果特征
        # 此处的乘法保留了梯度流，使得下游的 cls_loss 和 got_loss
        # 可以正常向回传播并指导 h（即要求 h 把有用的物理信息集中在未被掩码屏蔽的通道中）
        f_causal = h * causal_mask

        # 掩码互斥性与稀疏性约束（现仅约束 mask_generator 自身的权重）
        indep_loss = torch.mean(causal_mask * (1.0 - causal_mask))
        mask_mean = torch.mean(causal_mask)
        sparsity_loss = torch.abs(mask_mean - 0.5)

        # 联合解耦损失
        total_causal_loss = indep_loss + 0.1 * sparsity_loss

        return f_causal, total_causal_loss

