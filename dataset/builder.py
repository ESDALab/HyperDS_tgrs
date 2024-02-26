import sys
sys.path.append('/mnt/petrelfs/liuzili/code/DA_RS')
from dataset.dars_inf_downscale_dataset_mp import DARS_INF_Downscale_Dataset
import yaml
import xarray as xr
from easydict import EasyDict

dataset_dict = {
                'DARS_INF_Downsacle_Dataset': DARS_INF_Downscale_Dataset
                }

def build_dataset(name='DARS_INF_Downsacle_Dataset', config = None, type = None):
    if name in dataset_dict.keys():
        return dataset_dict[name](config, type)
    else:
        raise NotImplementedError(r"{} is not an avaliable values in dataset_dict. ".format(name))


if __name__ == '__main__':
    config_path = '/mnt/petrelfs/liuzili/code/DA_RS/configs/config_cross_unet.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)
    dataset = build_dataset(config.dataset_name, config.train_cfg, 'train')
