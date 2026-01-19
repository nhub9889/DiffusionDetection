import torch
import torch.nn as nn
import timm
from torchvision.ops import FeaturePyramidNetwork
import yaml

class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        backbone_cfg = cfg['backbone']
        neck_cfg = cfg.get('neck', None)

        self.backbone = timm.create_model(
            backbone_cfg['name'],
            pretrained= backbone_cfg['pretrained'],
            features_only= True,
            out_indices=backbone_cfg['out_indices'],
        )

        if backbone_cfg.get('frozen_stage', -1) >= 0:
            self._freeze_stages(backbone_cfg['frozen_stage'])
            feature_info = self.backbone.feature_info
            in_channels = [x['num_chs'] for x in feature_info]

            print(f"Backbone '{backbone_cfg['name']}' has {in_channels}")
            self.neck = None

            if neck_cfg and neck_cfg['name'] == 'fpn':
                self.neck = FeaturePyramidNetwork(
                    in_channels_list=in_channels,
                    out_channels=neck_cfg['out_channels']
                )

    def _freeze_stages(self, frozen_stages):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)
        features_dict = {str(i): f for i, f in enumerate(features)}

        if self.neck:
            fpn_features = self.neck(features_dict)
            return list(fpn_features.values())

        return features


def build_model(yaml_path):
    with open(yaml_path, 'r') as f:
        full_config = yaml.safe_load(f)

    return Encoder(full_config['model'])