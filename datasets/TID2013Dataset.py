import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize, RandomHorizontalFlip, RandomCrop
import csv
import os
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from skimage.color import rgb2gray
from datasets.processing import error_map, low_frequency_sub
import scipy.misc as m



def TID2013_GTtable(gt_file):
    table = []
    score_all = []
    # with open(gt_file) as f:
    #     lines = f.readlines()
    #
    # for line in lines:
    #     score = float(line.split(' ')[0])
    #     img_name = line.split(' ')[1][:-1]
    #     table.append([img_name, score])
    #     score_all.append(score)

    with open(gt_file, newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            score = float(row[3])
            img_name = row[0]
            img_name = img_name[img_name.rfind('/')+1:]
            table.append([img_name, score])
            score_all.append(score)

    score_all = np.array(score_all)
    score_min = np.min(score_all)
    score_max = np.max(score_all - score_min)

    return np.array(table), score_min, score_max


class TID2013Dataset(Dataset):
    def __init__(self, csv_path, data_path, transforms=None, is_train=True, return_type='dist', test_data='live'):
        super().__init__()
        self.transforms = transforms
        self.transforms = transforms
        self.is_train = is_train
        self.return_type = return_type  # all, dist, ref
        self.p = 0.5

        # self.csv_path = 'C:\\Users\\yunjeongyong\\Desktop\\DeepQA-yunjeong\\data\\all_data_csv\\TID2013.txt.csv'
        self.csv_path = csv_path
        self.data_path = data_path
        self.data_tmp, self.dist_img_path, self.dist_type, self.ref_img_path, self.dmos = self.csv_read(self.csv_path)

        self.x_train, self.x_test, self.y_train, self.y_test = self.train_test_split(self.data_tmp, self.dmos, test_size=0.2, random_state=2, shuffle=True)

    def __getitem__(self, idx):
        if self.is_train:
            dist_img, ref_img = self.img_read(self.data_path, self.x_train[idx][0], self.x_train[idx][2])
            error_img, dist_img = self.cal_error(dist_img, ref_img)
            dist_img, ref_img, error_img = self.to_tensor(dist_img, ref_img, error_img)
            return dist_img, ref_img, error_img, self.y_train[idx]
        else:
            dist_img, ref_img = self.img_read(self.data_path, self.x_test[idx][0], self.x_test[idx][2])
            error_img, dist_img = self.cal_error(dist_img, ref_img)
            dist_img, ref_img, error_img = self.to_tensor(dist_img, ref_img, error_img)
            return dist_img, ref_img, error_img, self.y_test[idx]

    def __len__(self):
        if self.is_train:
            return len(self.y_train)
        else:
            return len(self.y_test)

    def train_test_split(self, X, y, test_size, random_state, shuffle):
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=shuffle
        )
        return x_train, x_test, y_train, y_test

    def csv_read(self, csv_path):
        dist_img_path = []
        dist_type = []
        ref_img_path = []
        dmos = []
        data_tmp = []
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                data_tmp.append(row[0:3])
                dist_img_path.append(row[0])
                dist_type.append(row[1])
                ref_img_path.append(row[2])
                dmos.append(float(row[3]))
            # rows = [row for row in reader]
        return data_tmp, dist_img_path, dist_type, ref_img_path, dmos

    def img_read(self, data_path, dist, ref):
        # dist_img = Image.open(data_path + dist)
        # dist_img = dist_img.convert("RGB")
        dist_img = m.imread(data_path + dist)

        # ref_img = Image.open(data_path + ref)
        # ref_img = ref_img.convert("RGB")
        ref_img = m.imread(data_path + ref)
        return dist_img, ref_img

    def cal_error(self, dist_img, ref_img):
        newsize = (112, 112)
        # dist_img = dist_img.resize(newsize)
        # ref_img = ref_img.resize(newsize)
        # dist_img = np.array(dist_img, dtype=np.float64)
        # ref_img = np.array(ref_img, dtype=np.float64)
        # dist_img = np.resize(dist_img, (112, 112, 3))
        # ref_img = np.resize(ref_img, (112, 112, 3))


        # dist_img = Image.fromarray(dist_img)
        # ref_img = Image.fromarray(ref_img)

        dist_img = m.imresize(dist_img, newsize, interp='nearest')
        ref_img = m.imresize(ref_img, newsize, interp='nearest')

        dist_gray = rgb2gray(dist_img)
        ref_gray = rgb2gray(ref_img)
        # dist_gray = rgb2gray(np.array(dist_img, dtype=np.float64))
        # ref_gray = rgb2gray(np.array(ref_img, dtype=np.float64))

        img_d = low_frequency_sub(dist_gray)
        img_r = low_frequency_sub(ref_gray)
        error_img = error_map(img_d, img_r, epsilon=1.)
        error_img_3d = np.expand_dims(error_img, axis=0)
        return error_img, img_d

    def to_tensor(self, dist_img, ref_img,err_img):
        # dist_img = Image.fromarray(dist_img)
        # ref_img = Image.fromarray(ref_img)

        if self.is_train == True:
            if np.random.rand() < self.p:
                dist_img = np.flip(dist_img, axis=1).copy()
                err_img = np.flip(err_img, axis=1).copy()

        totensor = ToTensor()
        dist_img = totensor(dist_img)
        ref_img = totensor(ref_img)
        # dist_img = dist_img.type(torch.cuda.FloatTensor)
        # dist_img = rgb2gray(dist_img)
        # ref_img = rgb2gray(ref_img)
        err_img = torch.from_numpy(err_img).float()
        # if self.is_train == True:
        #     if torch.rand(1) < self.p:
        #         dist_img = F.hflip(dist_img)
        #         err_img = F.hflip(err_img)

        # dist_img = totensor(dist_img)
        # ref_img = totensor(ref_img)
        # err_img = torch.from_numpy(err_img).float()
        # err_img = torch.as_tensor(np.array(err_img).astype('float'))


        # resizer = Resize((112, 112))
        # dist_img_t = resizer(dist_img)
        # ref_img_t = resizer(ref_img)
        # error_img_t = resizer(error_img)
        return dist_img, ref_img, err_img



if __name__ == "__main__":
    csv_path = 'C:\\Users\\yunjeongyong\\Desktop\\DeepQA-yunjeong\\data\\all_data_csv\\TID2013.txt.csv'
    data_path = 'C:\\Users\\yunjeongyong\\Desktop\\DeepQA-yunjeong\\data\\TID2013_dataset'
    dataset = TID2013Dataset(csv_path, data_path, transforms=None, is_train=True, return_type='all', test_data='tid')
    # dist_img, ref_img, error_img, dmos = dataset[10]
    for i in range(10):
        dist_img, ref_img, error_img, dmos = dataset[i]
    print(0)
