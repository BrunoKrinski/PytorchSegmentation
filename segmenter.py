import os
import cv2
import yaml
import torch
import shutil
import argparse
import numpy as np
import albumentations as albu
import segmentation_models_pytorch as smp

from dataset import *
from tqdm import trange

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

    torch.cuda.set_device(gpu)
    print(project_path)

    masks_path = project_path + '/predicted_masks'
    images_path = project_path + '/segmented_images'

    if os.path.isdir(masks_path):
        shutil.rmtree(masks_path)

    if os.path.isdir(images_path):
        shutil.rmtree(images_path)

    os.mkdir(masks_path)
    os.mkdir(images_path)
        
    model_path = project_path + '/checkpoints/last.pth'

    model = torch.load(model_path)
                    
    colors = []
    colors_path = 'colors.txt'
    with open(colors_path, 'r') as colors_file:
        colors = colors_file.read().splitlines()

    colors_str = colors[0:num_classes]

    colors = []
    for c in colors_str:
        color = []
        c = c.split(',')
        for item in c:
            color.append(int(item))
        colors.append(color)
    
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, 'imagenet')

    test_dataset = Dataset(configs['dataset']['test'], num_classes,
                           augmentation=get_validation_augmentation(resize_height, resize_width),
                           preprocessing=get_preprocessing(preprocessing_fn), mode='eval')

    for i in trange(len(test_dataset)):
        image, gt_mask, image_path, mask_path = test_dataset[i]
        idx = image_path.replace('.jpg','').split('/')[-1]
        
        x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
        pr_mask = model.predict(x_tensor)
        
        pr_mask = torch.argmax(pr_mask, dim=1)
        pr_mask = (pr_mask.squeeze().cpu().numpy())
        
        original_image = cv2.imread(image_path)
        original_height, original_width, _ = original_image.shape

        if (resize_height > resize_width):
            max_size = resize_height
        else:
            max_size = resize_width

        transform = albu.LongestMaxSize(max_size, interpolation=cv2.INTER_NEAREST, p=1)
        image = transform(image=original_image)['image']
            
        image_height, image_width, _ = image.shape
        transform = albu.CenterCrop(height=image_height, width=image_width, p=1)
        pr_mask = transform(image=pr_mask)['image']

        if (original_height > original_width):
            max_size = original_height
        else:
            max_size = original_width

        transform = albu.LongestMaxSize(max_size, interpolation=cv2.INTER_NEAREST, p=1)
        pr_mask = transform(image=pr_mask)['image']
            
        mask_height, mask_width = pr_mask.shape
            
        if mask_width < original_width:
            transform = albu.Resize(height=original_height,width=original_width, interpolation=cv2.INTER_NEAREST, p=1)
            pr_mask = transform(image=pr_mask)['image']
        
        elif mask_width > original_width:
            transform = albu.CenterCrop(height=original_height,width=original_width, p=1)
            pr_mask = transform(image=pr_mask)['image']
            
        mask_height, mask_width = pr_mask.shape
        final_mask = np.zeros((mask_height, mask_width, 3))
        final_mask[:,:,0] = pr_mask
        final_mask[:,:,1] = pr_mask
        final_mask[:,:,2] = pr_mask
        
        for unique_value in np.unique(pr_mask):
            final_mask = np.where(final_mask == [unique_value, unique_value, unique_value], colors[unique_value], final_mask) 
            
        image = original_image.astype('float32')
        final_mask = final_mask.astype('float32')

        pred_image = cv2.addWeighted(image, 0.9, final_mask, 0.8, 0.0)

        image_path = images_path + '/{}.jpg'.format(idx)
        mask_path = masks_path + '/{}.png'.format(idx)

        cv2.imwrite(image_path, pred_image)
        cv2.imwrite(mask_path, final_mask)