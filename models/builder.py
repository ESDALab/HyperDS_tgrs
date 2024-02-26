from models.inf_net import INF_Net
from models.compare_model.unet import UNet
from models.compare_model.edsr import EDSR
from models.compare_model.rcan import RCAN

import yaml
import xarray as xr
from easydict import EasyDict

model_dict = {
              'INF_Net': INF_Net,
              'UNet': UNet,
              'EDSR': EDSR,
              'RCAN': RCAN
                }

def build_model(name='CrossAttentionUNet', network_config=None):
    if name in model_dict.keys():
        return model_dict[name](**network_config)
    else:
        raise NotImplementedError(r"{} is not an avaliable values in dataset_dict. ".format(name))
