import os
import yaml
import time
import torch
import argparse
import albumentations as albu
import segmentation_models_pytorch as smp

from dataset import *
from datetime import datetime
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
    decoder = configs['model']['decoder']
    dataset = configs['general']['dataset']
    resize_width = configs['model']['width']
    resize_height = configs['model']['height']
    batch_size = configs['model']['batch_size']
    num_epochs = configs['model']['num_epochs']
    experiment = configs['general']['experiment']
    epoch_decay = configs['general']['epoch_decay']
    num_workers = configs['general']['num_workers']
    learning_rate = configs['model']['learning_rate']
    augmentations = configs['augmentation']['augmentations']
    augmentation_prob = configs['augmentation']['augmentation_prob']

    torch.cuda.set_device(gpu)
    
    runs_dir = 'RUNS'
    os.makedirs(runs_dir, exist_ok='True')

    runs_dir += '/' + experiment
    os.makedirs(runs_dir, exist_ok='True')

    runs_dir += '/' + dataset
    os.makedirs(runs_dir, exist_ok='True')

    runs_dir += '/' + decoder
    os.makedirs(runs_dir, exist_ok='True')

    runs_dir += '/' + encoder
    os.makedirs(runs_dir, exist_ok='True')
        
    out_dir = args.configs.replace('.yml','').split('/')[-1]
    out_dir = runs_dir + '/' + out_dir
    
    date = str(datetime.now()).replace(' ', '_')
    #out_dir = out_dir + '_' + date.split('.')[0]
    #os.makedirs(out_dir)

    checkpoints = out_dir + '/checkpoints'
    os.makedirs(checkpoints) 

    os.system('cp {} {}'.format(args.configs, out_dir))
    os.system('cp {} {}'.format('trainer.py', out_dir))

    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, 'imagenet')
    train_dataset = Dataset(configs['dataset']['train'], 
                            num_classes,
                            augmentation=get_training_augmentation(augmentations, 
                                                                   augmentation_prob, 
                                                                   resize_height, 
                                                                   resize_width),
                            preprocessing=get_preprocessing(preprocessing_fn))

    valid_dataset = Dataset(configs['dataset']['valid'], 
                            num_classes,
                            augmentation=get_validation_augmentation(resize_height, 
                                                                     resize_width),
                            preprocessing=get_preprocessing(preprocessing_fn))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    activation = 'softmax2d'        

    if decoder == 'fpn':
        model = smp.FPN(encoder_name=encoder, classes=num_classes, encoder_weights='imagenet', activation=activation)
    elif decoder == 'unet':
        model = smp.Unet(encoder_name=encoder, classes=num_classes, encoder_weights='imagenet', activation=activation)
    elif decoder == 'unetplusplus':
        model = smp.UnetPlusPlus(encoder_name=encoder, classes=num_classes, encoder_weights='imagenet', activation=activation)
    elif decoder == 'linknet':
        model = smp.Linknet(encoder_name=encoder, classes=num_classes, encoder_weights='imagenet', activation=activation)
    elif decoder == 'pspnet':
        model = smp.PSPNet(encoder_name=encoder, classes=num_classes, encoder_weights='imagenet', activation=activation)
    elif decoder == 'pan':
        model = smp.PAN(encoder_name=encoder, classes=num_classes, encoder_weights='imagenet', activation=activation)
    elif decoder == 'manet':
        model = smp.MAnet(encoder_name=encoder, classes=num_classes, encoder_weights='imagenet', activation=activation)
    elif decoder == 'deeplabv3':
        model = smp.DeepLabV3(encoder_name=encoder, classes=num_classes, encoder_weights='imagenet', activation=activation)
    elif decoder == 'deeplabv3plus':
        model = smp.DeepLabV3Plus(encoder_name=encoder, classes=num_classes, encoder_weights='imagenet', activation=activation)
                
    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.9),
    ]

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=learning_rate),
    ])

    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=learning_rate),])

    train_epoch = smp.utils.train.TrainEpoch(model, loss=loss, metrics=metrics, optimizer=optimizer, device=device, verbose=True)

    valid_epoch = smp.utils.train.ValidEpoch(model, loss=loss, metrics=metrics, device=device, verbose=True)

    logs = {}
    logs['train'] = []
    logs['valid'] = []
        
    max_fscore = 0
    max_iscore = 0
    patience = 10
    patience_cont = 0
    min_loss = 1000
    
    for i in range(num_epochs):
        print('\nEpoch: {}'.format(i))
        print('\nLearning Rate: {}'.format(optimizer.param_groups[0]['lr']))
        train_init = time.time()
        train_logs = train_epoch.run(train_loader)
        train_end = time.time()
        train_time = (train_end - train_init)

        valid_init = time.time()
        valid_logs = valid_epoch.run(valid_loader)
        valid_end = time.time()
        valid_time = (valid_end - valid_init)
        
        valid_loss = valid_logs["dice_loss"]
        if min_loss > valid_loss:
            min_loss = valid_loss
            print('Weight saved!')
            torch.save(model, '{}/last.pth'.format(checkpoints))
            patience_cont = 0
        else:
            patience_cont += 1
            
        if patience_cont >= patience:
            print('Ending training at epoch: ' + str(i))
            break

        if i > 0 and (i % epoch_decay == 0):
            print('Learning rate decreased!')
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 1