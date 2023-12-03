import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, random_split
import sklearn.preprocessing as skp
import sklearn.model_selection as skms
from torchvision.transforms import transforms
import torch.nn.functional as F

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

    def __init__(self, root_path, resize=(1, 20)):
        self.data = []
        self.targets = []
        row_num = resize[0]
        col_num = resize[1]

        df = pd.read_csv(root_path)
        for row in range(len(df)):
            single_data = torch.zeros(row_num, col_num)
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
            single_data = torch.Tensor(single_data)
            self.data.append(single_data)
            self.targets.append(single_label)

    def __getitem__(self, index):
        x = self.data[index]
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


class my_dataset_more_features(Dataset):
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

    def __init__(self, root_path, resize=40):
        self.num_features = 40
        self.data = []
        self.targets = []

        df = pd.read_csv(root_path)
        for row in range(len(df)):
            single_data = df.iloc[row, 19: 19 + self.num_features]
            single_label = df.loc[row, 'label']
            single_data = torch.Tensor(single_data)
            single_data = single_data.view(1, self.num_features)
            # 将(1, 40)形状数据转化为(1, resize, resize)形状数据
            # 创建一个全为 0 的张量，其形状为 (1, resize-40)
            zeros = torch.zeros(1, resize - self.num_features)
            # 将 zeros 连接到 single_data 上，以将其形状改变为 (1, 224)
            single_data = torch.cat((single_data, zeros), dim=1)
            single_data = single_data.view(1, resize, 1)
            single_data = single_data.repeat(1, 1, resize)
            self.data.append(single_data)
            self.targets.append(single_label)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.classes[self.targets[index]]
        return x, y

    def __len__(self):
        return len(self.data)


class GTZANDataset_more_features:
    def __init__(self, rootDir=r"..\dataset\archive\Data\features_3_sec.csv", resize=40):
        self.rootDir = rootDir

        self.data = my_dataset_more_features(rootDir, resize)

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

class GTZANDataset_scale:
    def __init__(self, rootDir=r"..\..\dataset\archive\Data\features_3_sec.csv", resize=40):
        df = pd.read_csv(rootDir)
        df = df.drop(labels="filename", axis=1)
        # Feature Extraction
        class_list = df.iloc[:, -1]
        converter = skp.LabelEncoder()
        y = converter.fit_transform(class_list)

        # Scale the features
        scaler = skp.StandardScaler()
        X = scaler.fit_transform(np.array(df.iloc[:, :-1], dtype=float))
        # resize X to (len(X), resize, resize), 多出来的部分用0填充
        X = torch.tensor(X)
        X = X.view(len(X), 1, -1)
        for i in range(len(X)):
            """
            Traceback (most recent call last):
            File "D:\yk\Study\2023fall\IntroToAI\project\example\src\yk_train\process_data_simple.py", line 190, in <module>
            dataset = GTZANDataset_scale(r"D:\yk\Study\2023fall\IntroToAI\project\example\dataset\archive\Data\features_3_sec.csv", resize=224)
            File "D:\yk\Study\2023fall\IntroToAI\project\example\src\yk_train\process_data_simple.py", line 165, in __init__
            X[i] = torch.cat((X[i], zeros), dim=1)
            RuntimeError: The expanded size of the tensor (58) must match the existing size (224) at non-singleton dimension 1.  Target sizes: [1, 58].  Tensor sizes: [224]
            未知错误，暂时不知道怎么解决
            """

            zeros = torch.zeros(1, resize - X.shape[2])
            print(X[i].shape)
            print(zeros.shape)
            X[i] = torch.cat((X[i], zeros), dim=1)
            X[i] = X[i].view(1, resize, 1)
            X[i] = X[i].repeat(1, 1, resize)

        # Split the dataset
        X_train, X_test, y_train, y_test = skms.train_test_split(X, y, test_size=0.3, random_state=0)
        X_train = torch.Tensor(X_train)
        X_test = torch.Tensor(X_test)
        y_train = torch.Tensor(y_train)
        y_test = torch.Tensor(y_test)
        self.trainDataset = torch.utils.data.TensorDataset(X_train, y_train)
        self.testDataset = torch.utils.data.TensorDataset(X_test, y_test)

    def __call__(self, train="False"):
        """
        :param train:
        :return: dataset
        """
        if train == "True":
            return self.trainDataset
        elif train == "False":
            return self.testDataset


if __name__ == "__main__":
    dataset = GTZANDataset_scale(r"D:\yk\Study\2023fall\IntroToAI\project\example\dataset\archive\Data\features_3_sec.csv", resize=224)
    print(dataset.trainDataset[0][0].shape)
