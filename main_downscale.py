import argparse
import os
import yaml
from easydict import EasyDict
from dars_downscale_dist import DataAssimilationRemoteSensingDist
from dars_downscale_dist_gird import DataAssimilationRemoteSensingDistGrid

def parse_args():
    parser = argparse.ArgumentParser(
        description = 'Pytorch implementation of Data Assimulation with Remote Sensing by Edward Liu'
    )
    parser.add_argument('--config', default = '/mnt/petrelfs/liuzili/code/OBDS/configs/train_downscale_hyperds.yaml')
    parser.add_argument('--grid', action='store_true')
    parser.add_argument('--infer', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    for k, v in vars(args).items():
        config[k] = v
    config = EasyDict(config)
    

    if config.grid:
        dars_operator = DataAssimilationRemoteSensingDistGrid(config)
    else:
        dars_operator = DataAssimilationRemoteSensingDist(config)
    
    if config.infer:
        dars_operator.inference()
    else:
        dars_operator.train()

if __name__ == '__main__':
    main()
    print("FIN")