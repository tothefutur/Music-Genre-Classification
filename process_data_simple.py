import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, random_split
import torchvision.transforms

class my_dataset(Dataset):

    classes = {'blues': 0,
               'classical': 1,
               'country': 2,
               'disco': 3,
               'hiphop': 4,
               'jazz': 5,
               'metal': 6,
               'pop': 7,
               'reggae': 8,
               'rock': 9}

    def __init__(self, root_path, resize=(1, 20)): #直接读取root_path下的csv文件，
        self.data = []
        self.targets = []
        row_num = resize[0]
        col_num = resize[1]
        #transform1 = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        df = pd.read_csv(root_path)
        for row in range(len(df)):
            single_data = torch.zeros(1,row_num, col_num)
            single_label = df.loc[row, 'label']
            for count in range(1, col_num + 1):
                mean_string_list = ['mfcc']
                mean_string_list.append(str(count))
                mean_string_list.append('_mean')
                mean_col = ''.join(mean_string_list)

                var_string_list = ['mfcc']
                var_string_list.append(str(count))
                var_string_list.append('_var')
                var_col = ''.join(var_string_list)
                mean = torch.tensor(df.loc[row, mean_col])
                var = torch.tensor(df.loc[row, var_col])
                single_data[:, count - 1] = torch.tensor(np.random.normal(mean, pow(var, 0.5), size=row_num))
            #single_data = transform1(single_data)
            #single_data = single_data.permute(0,3,1,2)
            #print(single_data.shape)
            self.data.append(single_data)
            self.targets.append(single_label)


    def __getitem__(self, index):
        x = self.data[index]
        #print(x.shape)
        y = self.classes[self.targets[index]]
        return x, y

    def __len__(self):
        return len(self.data)


class GTZANDataset:
    def __init__(self, rootDir=r"..\dataset\archive\Data\features_3_sec.csv", resize=(1, 20)):
        self.rootDir = rootDir

        self.data = my_dataset(rootDir, resize)

        self.trainDataset, self.testDataset = random_split(
            dataset=self.data,
            lengths=[int(len(self.data) * 0.7), len(self.data) - int(len(self.data) * 0.7)],
            generator=torch.Generator().manual_seed(0)
        )

    def __call__(self, train="False"):
        """
        :param train:
        :return: dataset (every picture is transformed into tensor whose size is [13, 432])
        """
        if train == "True":
            return self.trainDataset
        elif train == "False":
            return self.testDataset

if __name__ == "__main__":
    dataset = GTZANDataset(r"..\data\music\features_3_sec.csv", resize=(16, 16)).data
    x,y = dataset[1]
    print(x.shape)
    print(dataset[1])

