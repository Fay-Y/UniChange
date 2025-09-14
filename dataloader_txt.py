import numpy as np
import os
from torch.utils.data import Dataset, DataLoader    
import numpy as np
import pickle
import random

class CCDataset(Dataset):
    def __init__(self, data_folder, split='train'):
        self.label_file = os.path.join(data_folder, f'{split}_labels.pkl') 
        self.image_file = os.path.join(data_folder, f'{split}_images.pkl') 
        self.caption_dir = os.path.join(data_folder, 'captions')  # caption txt目录
        self.split = split

        with open(self.label_file, 'rb') as file_label:
            self.label_data = pickle.load(file_label)
        with open(self.image_file, 'rb') as file_feat:
            self.image_data = pickle.load(file_feat)
        self.image_list = list(self.image_data.keys())

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_key = self.image_list[idx]
        image_info = self.image_data[image_key]
        label_info = self.label_data[image_key]
        image_before = np.transpose(image_info['image_before'], (2, 0, 1))
        image_after = np.transpose(image_info['image_after'], (2, 0, 1))

        feat_before = image_info['feat_before']
        feat_after = image_info['feat_after']
        label = np.transpose(label_info['label'], (2, 0, 1))
        feat_label = label_info['feat_label']

        caption_path = os.path.join(self.caption_dir, image_key+'.txt')
        with open(caption_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        text_info = random.choice(lines) if lines else "no caption available"

        return image_before, image_after, feat_before, feat_after, label, feat_label, text_info, image_key

if __name__ == '__main__':
    data_folder = 'datasets'
    dataset = CCDataset(data_folder, split='test')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    i= 0
    for images_before, images_after,  feat_before, feat_after, labels,feat_label, texts, keys in dataloader:
        print(i+1, "Batch processed")
        print(images_before.shape,  feat_after.shape, labels.shape,feat_label.shape, [texts])
        i += 1
