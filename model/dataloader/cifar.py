import pickle
from this import s

from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import json

class Cifar(Dataset):
    """ Usage:
    """
    def __init__(self, setname, args, augment=False):
        print('loading cache {}...'.format(setname))
        path = '/mnt/hdd/yw/cifar-fs/CIFAR_FS_' +  setname + '.pickle'
        with open(path, 'rb') as fo:
            split = pickle.load(fo, encoding='latin1')
        self.data, labels = split['data'], split['labels']
        with open('/home/yunwei/new/data/cifar_label_map_'+ setname + '.json', 'r') as f:
            mapper = json.load(f)
        self.label = []
        for label in labels:
            self.label.append(mapper[str(label)])

        self.num_class = len(set(self.label))

        image_size = 84
        if augment and setname == 'train':
            transforms_list = [
                  transforms.ToPILImage(),
                  transforms.Resize(image_size),
                  transforms.RandomResizedCrop(image_size),
                  transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                ]
        else:
            transforms_list = [
                  transforms.ToPILImage(),
                  transforms.Resize(92),
                  transforms.CenterCrop(image_size),
                  transforms.ToTensor(),
                ]

        # Transformation
        if args.backbone_class == 'ConvNet':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])
        elif args.backbone_class == 'Res12':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([0.4914, 0.4822, 0.4465]),
                                     np.array([0.247, 0.243, 0.261]))
            ])
        elif args.backbone_class == 'Res18':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])            
        elif args.backbone_class == 'WRN':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])         
        else:
            raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, label = self.data[i], self.label[i] #, self.name[i]
        image = self.transform(data)
        return image, label, i
        