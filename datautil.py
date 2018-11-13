# dataloader
# dataset
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os.path,os
import torchvision.transforms as transforms


def default_loader(path):
    return Image.open(path).convert('RGB')


class SceneDataset(Dataset):
    def __init__(self, image_file_path,img_transform=None, loader=default_loader):
        self.train = os.path.isfile(image_file_path)
        if self.train:
            with open(image_file_path) as f:
                # headers
                f.readline()
                self.images = list(map(lambda x: x.strip().split(','), f))
            self.root = os.path.join(os.path.dirname(image_file_path), 'data')
        elif os.path.isdir(image_file_path):
            self.images = [i[:-4] for i in os.listdir(image_file_path)]
            self.root = image_file_path
        self.img_transform = img_transform
        self.loader = loader

    def __getitem__(self, item):
        if self.train:
            img, label = self.images[item]
        else:
            img = self.images[item]
        if self.img_transform is not None:
            img = self.img_transform(self.loader(os.path.join(self.root, img+'.jpg')))
        return img if not self.train else (img,int(label))

    def __len__(self):
        return len(self.images)


def split_train_dataset(list_file):
    import random
    with open(list_file, 'r') as f:
        header = f.readline()
        l = f.readlines()
        val_list = random.sample(l,int(0.1*len(l)))
        for _ in val_list:
            l.remove(_)
    with open(os.path.join(os.path.dirname(list_file),'train_list.txt'),'w') as f:
        f.writelines([header]+l)
    with open(os.path.join(os.path.dirname(list_file),'val_list.txt'),'w') as f:
        f.writelines([header]+val_list)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data', metavar='PATH',help='')
    args = parser.parse_args()
    split_train_dataset(args.data)
