import torch
import torch.nn as nn
import torch.nn.functional as F
from convs.CL.resnet_cifar import cl_res32_cifar


class CL_cifar(nn.Module):
    def __init__(self, cfg, mode="train"):
        super(CL_cifar, self).__init__()
        pretrain = (
            True
            if mode == "train"
            and cfg.RESUME_MODEL == ""
            and cfg.BACKBONE.PRETRAINED_MODEL != ""
            else False
        )

        # self.num_classes = num_classes
        self.cfg = cfg

        self.backbone = cl_res32_cifar(
            self.cfg,
            pretrain=pretrain,
            pretrained_model=cfg.BACKBONE.PRETRAINED_MODEL,
            last_layer_stride=2,
        )
        self.module = GAP()
        # self.classifier = self._get_classifer()
        self.out_dim = self.get_feature_length()
        self.head = nn.Linear(self.out_dim, self.out_dim//2)


    def forward(self, x, **kwargs):
        if "feature_CL" in kwargs:
            return self.extract_feature_cl(x)
        elif "feature_class" in kwargs:
            return self.extract_feature(x)

        x = self.backbone(x)
        x = self.module(x)
        x = x.view(x.shape[0], -1)

        # x = self.classifier(x)
        return x


    def extract_feature(self, x):
        x = self.backbone(x)
        x = self.module(x)
        x = x.view(x.shape[0], -1)
        return x

    def extract_feature_cl(self, x):
        x = self.backbone(x)
        x = self.module(x)
        x = x.view(x.shape[0], -1)
        x = F.normalize(self.head(x), dim=1)
        return x


    def freeze_backbone(self):
        print("Freezing backbone .......")
        for p in self.backbone.parameters():
            p.requires_grad = False


    def load_backbone_model(self, backbone_path=""):
        self.backbone.load_model(backbone_path)
        print("Backbone has been loaded...")


    def load_model(self, model_path):
        pretrain_dict = torch.load(
            model_path, map_location="cpu" if self.cfg.CPU_MODE else "cuda"
        )
        pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
        model_dict = self.state_dict()
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v
        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        print("Model has been loaded...")


    def get_feature_length(self):
        num_features = 512
        return num_features


class GAP(nn.Module):
    """Global Average pooling
        Widely used in ResNet, Inception, DenseNet, etc.
     """

    def __init__(self):
        super(GAP, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.avgpool(x)
        #         x = x.view(x.shape[0], -1)
        return x

def get_model(cfg, device):
    model = CL_cifar(cfg, mode="train")

    if cfg.BACKBONE.FREEZE == True:
        model.freeze_backbone()

    # if cfg.CPU_MODE:
    #     model = model.to(device)
    # else:
    #     model = torch.nn.DataParallel(model).cuda()

    return model
