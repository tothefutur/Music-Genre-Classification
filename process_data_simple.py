import torch
import pandas as pd
from torch.utils.data import Dataset, random_split

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
                var = torch.tensor((df.loc[row, var_col]))
                single_data[:, count - 1] = torch.normal(mean, pow(var, 0.5))
            single_data = torch.Tensor(single_data)
            self.data.append(single_data)
            self.targets.append(single_label)


    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
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
    dataset = GTZANDataset(r"..\dataset\archive\Data\features_3_sec.csv").data
    print(dataset[1])

