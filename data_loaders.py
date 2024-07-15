import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# import matplotlib.pyplot as plt

class PlainDataset(Dataset):
    def __init__(self, csv_file, img_dir, datatype, transform):
        """
        PyTorch Dataset class for loading data.

        Args:
            csv_file (str): Path to the CSV file containing labels.
            img_dir (str): Directory containing the images.
            datatype (str): Identifier for the type of data (e.g., 'train', 'val', 'test').
            transform (torchvision.transforms.Compose): Transformations to be applied to the images.
        """
        self.csv_file = pd.read_csv(csv_file)
        self.labels = self.csv_file['emotion']
        self.img_dir = img_dir
        self.transform = transform
        self.datatype = datatype

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_dir + self.datatype + str(idx) + '.jpg'
        img = Image.open(img_path)
        label = np.array(self.labels[idx])
        label = torch.from_numpy(label).long()

        if self.transform:
            img = self.transform(img)

        return img, label

# # Helper function
# def eval_data_dataloader(csv_file,img_dir,datatype,sample_number,transform= None):
#     '''
#     Helper function used to evaluate the Dataset class
#     params:-
#             csv_file : the path of the csv file    (train, validation, test)
#             img_dir  : the directory of the images (train, validation, test)
#             datatype : string for searching along the image_dir (train, val, test)
#             sample_number : any number from the data to be shown
#     '''
#     if transform is None :
#         transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
#     dataset = Plain_Dataset(csv_file=csv_file,img_dir = img_dir,datatype = datatype,transform = transform)

#     label = dataset.__getitem__(sample_number)[1]
#     print(label)
#     imgg = dataset.__getitem__(sample_number)[0]
#     imgnumpy = imgg.numpy()
#     imgt = imgnumpy.squeeze()
#     plt.imshow(imgt)
#     plt.show()
