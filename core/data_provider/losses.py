"""
Different loss functions
Loss functions and seeting up loss function
"""
import torch
import torch.nn.functional as F

class KLLoss():
    """ Kullback-Leibler loss """

    def __call__(self, slow, slow_pre, middle, middle_pre, fast, fast_pre):
        """
        Computing KL-loss for a minibatch #计算小批量的 KL 损失

        Args:
        -----
        mu1, mu2, logvar1, logvar2: lists
            Lists of lists containing the mean and log-variances for the prior and posterior distributions,
            where each element is a tensor of shape (B, *latent_dim) # 包含先验分布和后验分布的均值和对数方差的列表列表，其中每个元素都是形状为 (B, *latent_dim) 的张量
        # """
        # if (len(mu1) > 0 and (not isinstance(mu1[0], list) or len(mu1[0]) > 0)):
        #     if (isinstance(mu1[0], list)):  # HierarchModel case# True
        #         mu1, logvar1 = [torch.stack(m, dim=1) for m in mu1], [torch.stack(m, dim=1) for m in logvar1]
        #         mu2, logvar2 = [torch.stack(m, dim=1) for m in mu2], [torch.stack(m, dim=1) for m in logvar2]
        #         loss = 0.
        #         for m1, lv1, m2, lv2 in zip(mu1, logvar1, mu2, logvar2):
        #             kld = self._kl_loss(m1, lv1, m2, lv2)
        #             loss += kld.sum() / kld.shape[0]
        #     else:
        #         mu1, logvar1 = torch.stack(mu1, dim=1), torch.stack(logvar1, dim=1)  # stacking across Frame dim
        #         mu2, logvar2 = torch.stack(mu2, dim=1), torch.stack(logvar2, dim=1)  # stacking across Frame dim
        #         kld = self._kl_loss(mu1, logvar1, mu2, logvar2)
        #         loss = kld.sum() / kld.shape[0]
        # else:
        #     loss = torch.tensor(0.)
        loss = 0
        # for m1, lv1, m2, lv2 in zip(mu1, logvar1, mu2, logvar2):
        #     kl = F.kl_div(m1.softmax(dim=-1).log(), m2.softmax(dim=-1), reduction='sum')
        #     loss += kl
        for slow_item, slow_pre_item in zip(slow, slow_pre):
            kl = F.kl_div(slow_item.softmax(dim=-1).log(), slow_pre_item.softmax(dim=-1), reduction='sum')
            loss += kl
        for middle_item, middle_pre_item in zip(middle, middle_pre):
            kl = F.kl_div(middle_item.softmax(dim=-1).log(), middle_pre_item.softmax(dim=-1), reduction='sum')
            loss += kl
        for fast_item, fast_pre_item in zip(fast, fast_pre):
            kl = F.kl_div(fast_item.softmax(dim=-1).log(), fast_pre_item.softmax(dim=-1), reduction='sum')
            loss += kl

        return loss

    def _kl_loss(self, mu1, logvar1, mu2, logvar2):
        """ Computing the KL-Divergence between two Gaussian distributions """ # 计算两个高斯分布之间的 KL-散度
        sigma1 = logvar1.mul(0.5).exp()
        sigma2 = logvar2.mul(0.5).exp()
        kld = torch.log(sigma2/sigma1) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(2*torch.exp(logvar2)) - 1/2
        return kld
