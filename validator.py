import yaml
import time
import torch
import argparse
import numpy as np
import segmentation_models_pytorch as smp

from dataset import *
from torch.utils.data import DataLoader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, required='true')
    args = parser.parse_args()

    with open(args.configs) as configs_file:
        configs = yaml.load(configs_file, Loader=yaml.FullLoader)

    with open(configs['dataset']['labels'], 'r') as classes_file:
        classes = classes_file.read().splitlines()

    device = 'cuda'
    num_classes = len(classes)    
    
    gpu = configs['general']['gpu']
    encoder = configs['model']['encoder']
    resize_width = configs['model']['width']
    project_path = configs['general']['path']
    resize_height = configs['model']['height']
    batch_size = configs['model']['batch_size']
    num_workers = configs['general']['num_workers']

    torch.cuda.set_device(gpu)
    print(project_path)
        
    model_path = project_path + '/checkpoints/last.pth'

    model = torch.load(model_path)
    
    loss = smp.utils.losses.JaccardLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, 'imagenet')
    test_dataset = Dataset(configs['dataset']['test'], 
                           num_classes,
                           augmentation=get_validation_augmentation(resize_height, 
                                                                    resize_width),
                           preprocessing=get_preprocessing(preprocessing_fn))
        
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_epoch = smp.utils.train.ValidEpoch(model=model, loss=loss, metrics=metrics, device=device)
        
    test_init = time.time()
    test_logs = test_epoch.run(test_loader)
    test_end = time.time()
    test_time = (test_end - test_init)