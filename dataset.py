import cv2
import numpy as np
import albumentations as albu
import segmentation_models_pytorch as smp

from datetime import datetime
from torch.utils.data import Dataset as BaseDataset

class Dataset(BaseDataset):
    
    def __init__(self, ids, num_classes, have_mask=True, augmentation=None, preprocessing=None, mode=None):
        
        with open(ids, 'r') as ids_file:
            self.ids = ids_file.read().splitlines()
        
        self.images = []
        self.masks = []
        self.have_mask = have_mask

        if self.have_mask:
            for idx in self.ids:
                image_path, mask_path = idx.split(' ')
                self.images.append(image_path)
                self.masks.append(mask_path)
        else:
            for idx in self.ids:
                self.images.append(idx)

        self.num_classes = num_classes
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.mode = mode
    
    def __getitem__(self, i):
        
        image_path = self.images[i]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.have_mask:
            mask_path = self.masks[i]      
            mask = cv2.imread(mask_path, 0)
        
        height = image.shape[0]
        width = image.shape[1]

        masks = np.zeros((height, width, self.num_classes))

        if self.have_mask:
            for i, unique_value in enumerate(np.unique(mask)):
                masks[:, :, unique_value][mask == unique_value] = 1
                
        if self.augmentation:
            sample = self.augmentation(image=image, mask=masks)
            image, masks = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=masks)
            image, masks = sample['image'], sample['mask']

        if self.mode == 'eval':
            return image, masks, image_path, mask_path
        elif self.mode == 'test':
            return image, masks, image_path
        return image, masks
    
    def __len__(self):
        return len(self.ids)

def get_training_augmentation(augmentations, prob, height=256, width=256):
    if (height > width):
        max_size = height
    else:
        max_size = width
    
    train_transform = [
        albu.LongestMaxSize(max_size, interpolation=cv2.INTER_NEAREST, p=1),
        albu.PadIfNeeded(min_height=height, min_width=width, always_apply=True, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
    ]

    for augmentation in augmentations:
        if augmentation == 'clahe':
            train_transform.append(albu.CLAHE(p=prob))
        elif augmentation == 'emboss':
            train_transform.append(albu.Emboss(p=prob))
        elif augmentation == 'gaussian_blur':
            train_transform.append(albu.GaussianBlur(p=prob))
        elif augmentation == 'image_compression':
            train_transform.append(albu.ImageCompression(p=prob, quality_lower=70, quality_upper=100))
        elif augmentation == 'median_blur':
            train_transform.append(albu.MedianBlur(p=prob))
        elif augmentation == 'posterize':
            train_transform.append(albu.Posterize(p=prob))
        elif augmentation == 'random_brightness_contrast':
            train_transform.append(albu.RandomBrightnessContrast(p=prob))
        elif augmentation == 'random_gamma':
            train_transform.append(albu.RandomGamma(p=prob))
        elif augmentation == 'random_snow':
            train_transform.append(albu.RandomSnow(p=prob))
        elif augmentation == 'sharpen':
            train_transform.append(albu.Sharpen(p=prob))
        
        elif augmentation == 'coarse_dropout':
            train_transform.append(albu.CoarseDropout(p=prob, max_holes=200))
        elif augmentation == 'elastic_transform':
            train_transform.append(albu.ElasticTransform(p=prob, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0))
        elif augmentation == 'flip':
            train_transform.append(albu.Flip(p=prob))    
        elif augmentation == 'grid_distortion':
            train_transform.append(albu.GridDistortion(p=prob, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0))
        elif augmentation == 'grid_dropout':
            train_transform.append(albu.GridDropout(p=prob))
        elif augmentation == 'optical_distortion':
            train_transform.append(albu.OpticalDistortion(p=prob, distort_limit=0.2, shift_limit=0.2, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0))
        elif augmentation == 'piecewise_affine':
            train_transform.append(albu.PiecewiseAffine(p=prob, interpolation=1, mask_interpolation=1, cval=0, cval_mask=0, mode='constant'))
        elif augmentation == 'random_crop':
            train_transform.append(albu.RandomCrop(p=prob, width=max_size, height=max_size))
        elif augmentation == 'rotate':
            train_transform.append(albu.Rotate(p=prob, limit=180, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0))
        elif augmentation == 'shift_scale_rotate':
            train_transform.append(albu.ShiftScaleRotate(p=prob, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0))
        elif augmentation == 'noda':
            continue
        else:
            print('Unknown data augmentation')
            exit()
    print('Data Augmentations applied:')
    print(train_transform)
    return albu.Compose(train_transform)

def get_validation_augmentation(height=256, width=256):
    if (height > width):
        max_size = height
    else:
        max_size = width

    test_transform = [
        albu.LongestMaxSize(max_size, interpolation=cv2.INTER_NEAREST, p=1),
        albu.PadIfNeeded(min_height=height, min_width=width, always_apply=True, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
    ]
    return albu.Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor,mask=to_tensor)
    ]
    return albu.Compose(_transform)