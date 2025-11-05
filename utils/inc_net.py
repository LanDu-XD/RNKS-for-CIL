import copy
import torch
from torch import nn
from convs.cifar_resnet import resnet32
from convs.resnet import resnet18, resnet34, resnet50
from convs.CL.CL_cifar import get_model
from convs.linears import SimpleLinear, SplitCosineLinear, CosineLinear
from default import _C as cfg, update_config

update_config(cfg)

def get_convnet(convnet_type, pretrained=False):
    name = convnet_type.lower()
    if name == 'resnet32':
        return resnet32()
    elif name == 'resnet18':
        return resnet18(pretrained=pretrained)
    elif name == 'resnet34':
        return resnet34(pretrained=pretrained)
    elif name == 'resnet50':
        return resnet50(pretrained=pretrained)
    elif name == 'cosine_resnet18':
        return cosine_resnet18(pretrained=pretrained)
    elif name == 'cosine_resnet32':
        return cosine_resnet32()
    elif name == 'cosine_resnet34':
        return cosine_resnet34(pretrained=pretrained)
    elif name == 'cosine_resnet50':
        return cosine_resnet50(pretrained=pretrained)
    elif name == 'protocl':
        return get_model(cfg, 'cuda')
    else:
        raise NotImplementedError('Unknown type {}'.format(convnet_type))


class BaseNet(nn.Module):

    def __init__(self, convnet_type, device, pretrained):
        super(BaseNet, self).__init__()

        self.convnet = get_convnet(convnet_type, pretrained)
        self.fc = None
        self.device = device

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet(x, feature_class=True)

    def extract_vector_cl(self, x):
        return self.convnet(x, feature_CL=True)

    # def forward(self, x):
    #     x = self.convnet(x)
    #     out = self.fc(x['features'])
    #     '''
    #     {
    #         'fmaps': [x_1, x_2, ..., x_n],
    #         'features': features
    #         'logits': logits
    #     }
    #     '''
    #     out.update(x)
    #
    #     return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self


class IncrementalNet(BaseNet):

    def __init__(self, convnet_type, pretrained, gradcam=False):
        super().__init__(convnet_type, pretrained)
        self.gradcam = gradcam
        if hasattr(self, 'gradcam') and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x['features'])
        out.update(x)
        if hasattr(self, 'gradcam') and self.gradcam:
            out['gradcam_gradients'] = self._gradcam_gradients
            out['gradcam_activations'] = self._gradcam_activations

        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(backward_hook)
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(forward_hook)



class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x, low_range, high_range):
        ret_x = x.clone()
        ret_x[:, low_range:high_range] = self.alpha * x[:, low_range:high_range] + self.beta
        return ret_x

    def get_params(self):
        return (self.alpha.item(), self.beta.item())


class IncrementalNetWithPROTOCL(BaseNet):
    def __init__(self, convnet_type, device, epoch_number, pretrained):
        super().__init__(convnet_type, device, pretrained)
        self.cfg = cfg
        self.device = device
        self.epoch_number = epoch_number
        self.initilize_all_parameters()

    def initilize_all_parameters(self):
        self.alpha = 0.2
        if self.epoch_number in [90, 180]:
            self.div_epoch = 100 * (self.epoch_number // 100 + 1)
        else:
            self.div_epoch = self.epoch_number

    def reset_epoch(self, epoch):
        self.epoch = epoch

    def forward(self, image, meta):

        image_a, image_b = image.to(self.device), meta["sample_image"].to(self.device)


        feature_a, feature_b = (
            self.convnet(image_a, feature_CL=True),
            self.convnet(image_b, feature_class=True),
        )
        cl_features = feature_a

        l = 1 - ((self.epoch - 1) / self.div_epoch) ** 2  # parabolic decay

        out = self.fc(feature_b)

        return out, cl_features, l

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
