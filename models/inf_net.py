from models.pixel_net import PixelWiseAdpNet
from models.dual_transnet import DualTransNet
from models.variable_net import VariableNet
import torch
import torch.nn as nn
import torch.nn.functional as F

from numpy import pi

class INF_Net(nn.Module):
    def __init__(self,
                 dual_trans_cfg: dict,
                 inf_net_cfg: dict,
                 var_net_cfg: dict,
                 decoder_type='MULTI_VAR',
                 **kwargs):
        super().__init__()
        self.dual_transnet = DualTransNet(**dual_trans_cfg)
        self.inf_net_cfg = inf_net_cfg
        self.decoder_type = decoder_type
        self.target_wh = inf_net_cfg.target_wh
        self.out_keys = inf_net_cfg.out_keys
        self.out_channel = inf_net_cfg.out_channel
        if self.decoder_type == 'MULTI_BLOCK':
            self.inf_net1 = PixelWiseAdpNet(**(inf_net_cfg.inf_net_block4_cfg))
        elif self.decoder_type == 'MULTI_VAR':
            self.token_num = var_net_cfg.token_num
            self.in_channel = var_net_cfg.in_channel
            self.hidden_channel = var_net_cfg.hidden_channel
            self.U_net = VariableNet(self.token_num, self.in_channel, self.hidden_channel)
            self.V_net = VariableNet(self.token_num, self.in_channel, self.hidden_channel)
            self.T_net = VariableNet(self.token_num, self.in_channel, self.hidden_channel)
            self.P_net = VariableNet(self.token_num, self.in_channel, self.hidden_channel)
            self.TP_net = VariableNet(self.token_num, self.in_channel, self.hidden_channel)
        else:
            raise NotImplementedError

    
    
    def forward_single(self, MLP_feature, coord_em, coord_data, flag, device, **kwargs):
        if self.decoder_type == 'MULTI_BLOCK':
            out = self.inf_net1(MLP_feature, coord_em, coord_data, flag, device, **kwargs)
            return out
        elif self.decoder_type == 'MULTI_VAR':
            U = self.U_net(MLP_feature, coord_em, coord_data[:,:,0])
            V = self.V_net(MLP_feature, coord_em, coord_data[:,:,1])
            T = self.T_net(MLP_feature, coord_em, coord_data[:,:,2])
            P = self.P_net(MLP_feature, coord_em, coord_data[:,:,3])
            TP = self.TP_net(MLP_feature, coord_em, coord_data[:,:,4])
            out = torch.cat([U,V,P,T,TP],dim=2)
            return out

    def forward(self, h8_data, pred_sfc_data, coord_em, coord_data, flag, device, **kwargs):
        # import pdb
        # pdb.set_trace()
        bSize = h8_data.size()[0]
        feature_dict = self.dual_transnet(h8_data, pred_sfc_data)
        for key in self.out_keys:
            feature = feature_dict[key]
            inf_cfg_str = 'inf_net_'+key+'_cfg'
            inf_cfg = self.inf_net_cfg[inf_cfg_str]
            
            out = self.forward_single(feature, coord_em, coord_data, flag, device, **kwargs)

        return out
    
