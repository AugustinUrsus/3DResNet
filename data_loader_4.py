import torch
import torchvision
import torch.utils.data as Data
import torchvision.transforms as Transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import os
import math
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image


#data_path = 'D:/Datasets/BU_3DFE/RGB'
#data_path_d = 'D:/Datasets/BU_3DFE/D'
classes_n = 0

class DataType():
    "Stores the paths to images for a given class"
    def __init__(self, name, rgb_paths, d_paths):
        self.class_name = name
        self.rgb_paths = rgb_paths
        self.d_paths = d_paths

class ImagePath():
    def __init__(self, rgb_path, d_path):
        self.rgb_path = rgb_path
        self.d_path = d_path
# modified to load jpg images only
def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = list(filter(lambda a: 'jpg' in a,[os.path.join(facedir,img) for img in images]))

    return image_paths

def get_all_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        path_rgb = list(filter(lambda a: 'jpg' in a,[os.path.join(facedir,img) for img in images]))
        path_d = list(filter(lambda a: os.path.exists(a), [img.replace('.jpg', '.npy') for img in path_rgb ])) # get existed npy paired with jpg files
        path_rgb = [a.replace('.npy','.jpg') for a in path_d] # generate rbg paths from depth pathes
    return path_rgb, path_d

def get_image_d_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = list(filter(lambda a: 'npy' in a,[os.path.join(facedir,img) for img in images]))

    return image_paths

def get_labels(data):
    images = []
    labels = []
    for i in range(len(data)):
#         print(i)
#         print(len(data[i].rgb_paths) - len(data[i].d_paths))
        images += [ImagePath(data[i].rgb_paths[j], data[i].d_paths[j]) for j in range(len(data[i].rgb_paths))]
        labels += [data[i].class_name] * len(data[i].rgb_paths)
    
    return np.array(images), np.array(labels)


def load_data(data_path, data_path_d):
    dataset = []
    classes = [path for path in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, path))]

    # upper limit for this dataset
    classes_n = 10000# len(classes)

    for i in range(classes_n):
        path = 'n{:06d}'.format(i+1)
        # check path existance
        if os.path.exists(path):
#             if i % 10 == 0:
#                 print(i+1)
            face_dir = os.path.join(data_path, path)
            #face_dir_d = os.path.join(data_path_d, path)

            # Get image pathes of this class
            image_paths, image_paths_d = get_all_image_paths(face_dir)# modified to read npy only
            if len(image_paths) == len(image_paths_d):
                assert(len(image_paths) == len(image_paths_d))
                dataset.append(DataType(i+1, image_paths, image_paths_d))


    train_x, train_y = get_labels(dataset)

    return classes_n, train_x, train_y



def my_loader(path, Type):
    #print(path)
    if Type == 3:
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')
    else:
#         print(type(path))
#         print(os.path.abspath(os.path.dirname(__file__)))
        try:        
            im = np.load(path).astype('uint8')

#         print(im.shape)
        except (Exception, TypeError, NameError):
            return None
        return im


class MyDataset(Data.Dataset):
    def __init__(self, img_paths, labels, transform, loader=my_loader):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
        self.loader = loader
    
    def __getitem__(self, index): #return data type is tensor
        rgb_path, d_path = self.img_paths[index].rgb_path, self.img_paths[index].d_path
        label = self.labels[index]
#         print(rgb_path, d_path, label)
        
        rgb_img = np.array( my_loader(rgb_path, 3) ,dtype=np.uint8) 
        d_img = my_loader(d_path, 1)

	# handle npy error
        if d_img is None:
            d_img = np.zeros((rgb_img.shape[0],rgb_img.shape[1]),dtype=np.uint8)
#         print(d_img.shape, rgb_img.shape)
        
        d_img = np.expand_dims(d_img, axis=2)

        

        img = np.append(rgb_img, d_img, axis=2)
        img = self.transform(Image.fromarray(img))
        label = torch.from_numpy(np.array(label)).type(torch.LongTensor)

        
        return img, label

    def __len__(self): # return the total size of the dataset
        return len(self.labels)



### Split dataset and creat train & valid dataloader ###
def split_dataset(dataset_t, batch, split_ratio):
    num_train = len(dataset_t)
    indices = list(range(num_train))
    split = int(np.floor(split_ratio * num_train))

    #np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(dataset_t, batch_size=batch, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(dataset_t, batch_size=batch, sampler=valid_sampler)

    return train_loader, valid_loader


def CreateDataloader(data_path, data_path_d, batch, split_ratio):
    classes_n, train_x, train_y = load_data(data_path, data_path_d)

    transform = Transforms.Compose([
        Transforms.Pad(90),
        Transforms.CenterCrop((224,224)),
        Transforms.Resize(224),
        Transforms.ToTensor(),
    ])
    
    dataset = MyDataset(train_x, train_y, transform=transform)

    train_loader, valid_loader = split_dataset(dataset, batch, split_ratio)

    print('Number of classes: %d' % classes_n)
    print('Total images: %d' % len(train_x))
    #print('Total images: %d (split ratio: %.1f)' % (len(train_x), split_ratio) )
    #print('Training images:', len(train_loader))
    #print('Validation images: ', len(valid_loader))

    return classes_n, train_loader, valid_loader
