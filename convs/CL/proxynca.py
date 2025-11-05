
import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn import init
import math
import numpy as np
import copy

def binarize_and_smooth_labels(T, nb_classes, smoothing_const = 1):
    import sklearn.preprocessing
    nb_classes = int(nb_classes)
    T = T.cpu().numpy()
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    # T = T * (1 - smoothing_const)
    # T[T == 0] = smoothing_const / (nb_classes - 1)
    T = torch.FloatTensor(T).cuda()

    return T


class ProxyNCA(torch.nn.Module):
    def __init__(self,
        device,
        sz_embedding,
        smoothing_const = 0,
        scaling_x = 1,
        scaling_p = 3
    ):
        torch.nn.Module.__init__(self)
        # initialize proxies s.t. norm of each proxy ~1 through div by 8
        # i.e. proxies.norm(2, dim=1)) should be close to [1,1,...,1]
        # TODO: use norm instead of div 8, because of embedding size
        self.proxies = None
        self.device = device
        self.nb_proxy = 6
        self.ratio = 0.1
        self.sz_embedding = sz_embedding
        self.smoothing_const = smoothing_const
        self.scaling_x = scaling_x
        self.scaling_p = scaling_p
        self.instance_label = None
        self.y_instacne_onehot = None
        self.func = torch.nn.Softmax(dim=1)
        self.weight_lambda = 0.3
        self.temperature = 0.35

    def to_one_hot(self, y, n_dims=None):
        ''' Take integer tensor y with n dims and convert it to 1-hot representation with n+1 dims. '''
        n_dims = int(n_dims)
        y_tensor = y.type(torch.LongTensor).view(-1, 1)
        y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
        y_one_hot = y_one_hot.view(*y.shape, -1)
        return y_one_hot

    def scale_mask_softmax(self,tensor,mask,softmax_dim,scale=1.0):
        #scale = 1.0 if self.opt.dataset != "online_products" else 20.0
        scale_mask_exp_tensor = torch.exp(tensor* scale) * mask.detach()
        scale_mask_softmax_tensor = scale_mask_exp_tensor / (1e-8 + torch.sum(scale_mask_exp_tensor, dim=softmax_dim)).unsqueeze(softmax_dim)
        return scale_mask_softmax_tensor

    def forward(self, X, T, mode='train'):
        cur_class = self.proxies.shape[-1]/self.nb_proxy
        #self.ratio = (0.5*cur_class+self.nb_proxy)/(cur_class*self.nb_proxy)

        P = F.normalize(self.proxies, p=2, dim=0)  # * self.scaling_p
        X = F.normalize(X, p=2, dim=-1)  # * self.scaling_x
        # constructing directed similarity graph
        similarity = X.matmul(P)
        # similarity = torch.div(similarity, self.temperature)
        # relation-guided sub-graph construction
        positive_mask = torch.eq(T.view(-1, 1).to(self.device) - self.instance_label.view(1, -1),
                                 0.0).float().to(self.device)
        topk = math.ceil(self.ratio * self.proxies.shape[-1])
        _, indices = torch.topk(similarity + 1000 * positive_mask, topk, dim=1)
        mask = torch.zeros_like(similarity)
        mask = mask.scatter(1, indices, 1)
        prob_a = mask * similarity
        # revere label propagation (including classification process)
        logits = torch.matmul(prob_a, self.y_instacne_onehot)
        y_target_onehot = binarize_and_smooth_labels(T, cur_class).to(self.device)
        logits_mask = 1 - torch.eq(logits, 0.0).float().to(self.device)
        predict = self.scale_mask_softmax(logits, logits_mask, 1).to(self.device)
        # classification loss
        lossClassify = torch.mean(torch.sum(-y_target_onehot * torch.log(predict + 1e-20), dim=1))
        if self.weight_lambda > 0 and mode == 'train':
            simCenter = P.t().matmul(P)
            centers_logits = torch.matmul(simCenter, self.y_instacne_onehot)
            reg = F.cross_entropy(centers_logits, self.instance_label)
            return lossClassify+self.weight_lambda*reg

    def update_proxy(self, nb_classes):
        proxies = self.generate_proxy(nb_classes)
        if self.proxies is not None:
            nb_output = self.proxies.shape[-1]
            proxies_old = copy.deepcopy(self.proxies.data)
            proxies.data[:,:nb_output] = proxies_old
        del self.proxies
        self.proxies = proxies
        self.instance_label = torch.tensor(np.repeat(np.arange(nb_classes), self.nb_proxy)).to(self.device)
        self.y_instacne_onehot = self.to_one_hot(self.instance_label, n_dims=nb_classes).to(self.device)

    def generate_proxy(self, nb_classes):
        proxies = Parameter(torch.Tensor(self.sz_embedding, nb_classes*self.nb_proxy))
        init.kaiming_uniform_(proxies, a=math.sqrt(5))

        return proxies

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
