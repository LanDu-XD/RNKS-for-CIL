from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


'''
CC with P-order Taylor Expansion of Gaussian RBF kernel
'''
class CC(nn.Module):
	'''
	Correlation Congruence for Knowledge Distillation
	http://openaccess.thecvf.com/content_ICCV_2019/papers/
	Peng_Correlation_Congruence_for_Knowledge_Distillation_ICCV_2019_paper.pdf
	'''
	def __init__(self, gamma, P_order, device):
		super(CC, self).__init__()
		self.gamma = gamma
		self.P_order = P_order
		self.device = device

	def forward(self, feat_s, feat_t, targets, known_classes):
		corr_mat_s = self.get_correlation_matrix(feat_s)
		corr_mat_t = self.get_correlation_matrix(feat_t)

		# loss = F.mse_loss(corr_mat_s, corr_mat_t)
		index = torch.gt(targets, (known_classes - 1)) + 0
		index = index.unsqueeze(1)
		index_matrix = index * index.T
		index_matrix = torch.tensor(index_matrix, dtype=torch.bool)
		mask = ~index_matrix + 0
		mask = torch.tensor(mask, dtype=torch.float32)
		mask = mask.to(self.device)
		corr_mat_s_final = mask * corr_mat_s
		corr_mat_t_final = mask * corr_mat_t
		loss = F.mse_loss(corr_mat_s_final, corr_mat_t_final)
		return loss

	def get_correlation_matrix(self, feat):
		feat = F.normalize(feat, p=2, dim=-1)
		sim_mat = torch.matmul(feat, feat.t())
		corr_mat = torch.zeros_like(sim_mat)

		for p in range(self.P_order+1):
			corr_mat += math.exp(-2*self.gamma) * (2*self.gamma)**p / \
						math.factorial(p) * torch.pow(sim_mat, p)

		return corr_mat
