import torch
import torch.nn as nn
from .backbone.builder import build_backbone
from .backbone.transformer_net import TransformerNet

import torch.nn.functional as F

class DualTransNet(nn.Module):
    def __init__(self, h8_backbone_cfg: dict,
                 pred_sfc_backbone_cfg: dict,
                 trans_cfg: dict,
                 head_cfg: dict, **kwargs):
        super().__init__()

        self.h8_backbone = build_backbone(**h8_backbone_cfg)
        self.pred_sfc_backbone = build_backbone(**pred_sfc_backbone_cfg)
        
        d_model = trans_cfg.d_model
        feat_channels = head_cfg.feat_channels
        self.img_size = head_cfg.img_size
        self.feat_size = head_cfg.feat_size

        token_num = 19 * 29
        pe_token = torch.rand([1, token_num, d_model])
        self.pred_conv3 = nn.Conv2d(feat_channels[1], d_model, kernel_size=3, stride=1, padding=1)
        self.h8_conv3 = nn.Conv2d(feat_channels[1]*2, d_model, kernel_size=3, stride=1, padding=1)
        self.pe_parameters3 = nn.Parameter(pe_token, requires_grad=True)
        self.transformer3 = TransformerNet(64, 64, 8, 1, 1, 64)

    def _forward_single(self, h8_feature, pred_feature, h8_conv, pred_conv, pe, transformer):
        
        h8_feature = h8_conv(h8_feature)
        pred_feature = pred_conv(pred_feature)
        B, C, H, W = h8_feature.shape
        
        pred_feature = F.interpolate(pred_feature, size=[H, W], mode='bilinear', align_corners=True)

        h8_x = h8_feature.view(B, C, -1)
        h8_x = torch.transpose(h8_x, 1, 2).contiguous()
        pred_x = pred_feature.view(B, C, -1)
        pred_x = torch.transpose(pred_x, 1, 2).contiguous()
        
        h8_x = h8_x + pe
        pred_x = pred_x + pe
        
        fuse_x = transformer(h8_x, pred_x)
        fuse_x = torch.reshape(torch.transpose(fuse_x, 1, 2).contiguous(), (B, C, H, W))

        return fuse_x

    def forward(self, h8_data, pred_sfc_data):        

        _, end_points0 = self.h8_backbone.forward(h8_data[:,0,:,:,:])
        _, end_points1 = self.h8_backbone.forward(h8_data[:,1,:,:,:])

        diff_feature = {}
        for key in end_points0.keys():
            diff_feature[key] = torch.cat((end_points0[key], end_points1[key]), dim=1)

        _, end_points = self.pred_sfc_backbone.forward(pred_sfc_data)
        
        features_dict = {}


        fuse_feature = self._forward_single(diff_feature['block4'],
                                            end_points['block4'],
                                            self.h8_conv3,
                                            self.pred_conv3,
                                            self.pe_parameters3,
                                            self.transformer3)
        features_dict['block4'] = fuse_feature

        return features_dict
        