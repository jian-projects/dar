import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class DotProductSimilarity(nn.Module):
    def __init__(self, scale_output=False) -> None:
        super(DotProductSimilarity, self).__init__()
        self.scale_output = scale_output
    
    def forward(self, tensor_1, tensor_2):
        result = torch.matmul(tensor_1, tensor_2.T)
        # result = (tensor_1*tensor_2).sum(dim=-1)
        if self.scale_output:
            result /= math.sqrt(tensor_1.size(-1))
        return result


class CosineSimilarity(nn.Module):
    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        super(CosineSimilarity, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return F.cosine_similarity(x1, x2, self.dim, self.eps)


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """
    def __init__(self, temp, method):
        super().__init__()
        self.temp = temp
        self.method = method
        self.cos = CosineSimilarity(dim=-1)
        self.dot = DotProductSimilarity()

    def forward(self, x, y):
        if self.method == 'cos':
            sim = self.cos(x, y) / self.temp
        if self.method == 'dot':
            sim = self.dot(x, y) / self.temp
        return sim 


class Contrast_Single(nn.Module):
    def __init__(self, method, temp=1) -> None:
        super().__init__()
        self.sim = Similarity(temp=temp, method=method)
        self.loss_ce = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, z1, z2, labels=None, loss=0):
        """
        仅有一个样本是目标样本的正样本

        z1_z2_sim -> [bz, samples_num] 
        labels -> [bz]
        
        labels确定哪个sample是正样本, 其他的均为负样本
        """
        
        z1_z2_sim = self.sim(z1.unsqueeze(dim=1), z2.unsqueeze(dim=0))
        # z1_z2_sim = self.sim(z1, z2)
        if labels is not None:
            loss = self.loss_ce(z1_z2_sim, labels)

        return z1_z2_sim

        return {
            'sim': z1_z2_sim,
            'loss': loss,
        }


class Contrast_Multi(Contrast_Single):
    def __init__(self, args, method) -> None:
        super().__init__(args, method)
        self.eps = 1e-12
        self.sim = Similarity(temp=5, method='dot')
        self.loss_ce = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, z1, z2, labels):
        """
        多个样本是目标样本的正样本
        z1: batch 样本表示 -> [bz, dim]
        z2: 对比样本表示 -> [num, dim] (batch样本表示拼接其他的/单独其他的)
        labels -> [bz, num] (1表示正样本, 0表示负样本) 
        
        labels 值为1的是对应的正样本, 其他的均为负样本
        """
        assert labels.shape == torch.Size([z1.shape[0], z2.shape[0]])
        z1_z2_sim = self.sim(z1, z2)
        z1_z2_sim -= z1_z2_sim.max(dim=-1)[0].detach() # 减去最大值？
        logits = torch.exp(z1_z2_sim)
        logits -= torch.eye(logits.shape[0]).type_as(logits)*logits.diag() # 去除自身
        log_prob = z1_z2_sim - torch.log(logits.sum(dim=1)+self.eps)
        log_prob_pos_mean = (labels*log_prob).sum(dim=1) / (labels.sum(dim=1)+self.eps)
        loss = (-log_prob_pos_mean).mean()

        return loss
        return {
            'loss': loss,
            'logits': log_prob,
        }