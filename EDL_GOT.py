# -*- coding:utf-8 -*-
# Usage: Implementation of Evidential Deep Learning (EDL) and Probability-Polarized Optimal Transport (PPOT).

import torch
import torch.nn.functional as F

def edl_loss(logits, target, epoch, total_epochs, num_classes):
    """
    证据深度学习 (EDL) 损失函数：施加受控的 KL 散度退火
    """
    evidence = F.softplus(logits)
    alpha = evidence + 1.0
    S = torch.sum(alpha, dim=1, keepdim=True)
    p = alpha / S

    y = F.one_hot(target, num_classes).float()

    err = torch.sum((y - p) ** 2, dim=1, keepdim=True)
    var = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)

    # 维持极小的 KL 正则化权重，保护非均衡数据集中的少数类流形
    annealing_coef = 0.01 * min(1.0, epoch / (total_epochs / 2.0))
    alp_tilde = (alpha - 1) * (1 - y) + 1.0

    kl_term = torch.lgamma(torch.sum(alp_tilde, dim=1, keepdim=True)) - \
              torch.sum(torch.lgamma(alp_tilde), dim=1, keepdim=True) + \
              torch.sum(torch.lgamma(torch.ones_like(alp_tilde)), dim=1, keepdim=True) - \
              torch.lgamma(torch.sum(torch.ones_like(alp_tilde), dim=1, keepdim=True)) + \
              torch.sum((alp_tilde - 1) * (
                          torch.digamma(alp_tilde) - torch.digamma(torch.sum(alp_tilde, dim=1, keepdim=True))), dim=1,
                        keepdim=True)

    return torch.mean(err + var + annealing_coef * kl_term)


def ppot_loss(x_s, x_t, y_s, logits_t, epoch, total_epochs, num_classes, epsilon=0.1, max_iter=50):
    """
    基于各向异性扩散算子与概率极化的几何最优传输 (PPOT) - 防坍缩修正版
    """
    B_s, B_t = x_s.size(0), x_t.size(0)

    with torch.no_grad():
        evidence_t = F.softplus(logits_t.detach())
        alpha_t = evidence_t + 1.0
        S_t = torch.sum(alpha_t, dim=1, keepdim=True)
        u_t = (num_classes / S_t).squeeze()
        pseudo_label_t = torch.argmax(alpha_t, dim=1)
        p_t = alpha_t / S_t

        # 课程学习的不确定性截断
        tau = 0.1 + 0.4 * (epoch / total_epochs)
        reliable_mask = u_t < tau
        unreliable_mask = ~reliable_mask

    # 【致密修复核心】：在度量几何流形距离前，强制 L2 特征归一化！
    # 将特征映射到单位超球面上，彻底剥夺网络通过将特征归 0 来作弊最小化代价矩阵的能力。
    x_s_norm = F.normalize(x_s, p=2, dim=1)
    x_t_norm = F.normalize(x_t, p=2, dim=1)

    # 1. 几何流形代价矩阵（计算球面特征的欧氏距离的各向异性核）
    dist_sq = torch.cdist(x_s_norm, x_t_norm, p=2) ** 2
    # 动态核带宽自适应
    sigma = torch.median(dist_sq).detach() + 1e-4
    # 将几何代价映射至严格的 [0, 1) 区间
    C_manifold = 1.0 - torch.exp(-dist_sq / (2 * sigma))

    C_polarized = C_manifold.clone()

    with torch.no_grad():
        y_s_expand = y_s.unsqueeze(1).expand(B_s, B_t)
        pred_t_expand = pseudo_label_t.unsqueeze(0).expand(B_s, B_t)

        # 2.1 正向极化惩罚 (仅针对满足严苛阈值的 reliable 样本)
        pos_penalty_mask = (y_s_expand != pred_t_expand) & reliable_mask.unsqueeze(0)

        # 2.2 负伪标签极化 (针对高不确定性样本，基于客观概率筛选)
        is_neg_class = p_t < 0.05
        is_source_class_neg = is_neg_class[:, y_s].t()
        neg_penalty_mask = is_source_class_neg & unreliable_mask.unsqueeze(0)

    # 在几何距离上施加势垒极化 (软极化上限 1.0)
    penalty_val = 1.0
    C_polarized[pos_penalty_mask] += penalty_val
    C_polarized[neg_penalty_mask] += penalty_val

    # 3. Sinkhorn-Knopp 最优传输求解
    K = torch.exp(-C_polarized / epsilon)
    u = torch.ones(B_s, device=x_s.device) / B_s
    v = torch.ones(B_t, device=x_t.device) / B_t

    for _ in range(max_iter):
        u = (1.0 / B_s) / (torch.matmul(K, v) + 1e-8)
        v = (1.0 / B_t) / (torch.matmul(K.t(), u) + 1e-8)

    pi = torch.diag(u) @ K @ torch.diag(v)
    ot_loss = torch.sum(pi * C_polarized)

    return ot_loss