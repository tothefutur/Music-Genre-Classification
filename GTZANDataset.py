import numpy as np
import librosa
import torch
from torch.utils.data import random_split
from torchvision import datasets


class PNGToMFCC(object):
    def __init__(self, n_mfcc=13):
        self.n_mfcc = n_mfcc

    def __call__(self, img):
        # 将PIL图像转换为灰度图像
        gray_image = img.convert('L')

        # 将灰度图像转换为数字数组
        mel_spec_array = np.array(gray_image, dtype=float)

        # 将数字数组转换为梅尔频谱图
        mel_spec = librosa.feature.inverse.mel_to_stft(mel_spec_array)

        # 将梅尔频谱图转换为对数刻度
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # 计算MFCC系数
        mfcc = librosa.feature.mfcc(S=log_mel_spec, n_mfcc=self.n_mfcc)

        # 选择前几个MFCC系数作为特征
        mfcc_features = mfcc[:self.n_mfcc]

        # 将特征矩阵转换为PyTorch张量
        mfcc_tensor = torch.from_numpy(mfcc_features)

        return mfcc_tensor.float()


class GTZANDataset:

    def __init__(self, rootDir=r"..//dataset//archive//Data//images_original"):
        self.rootDir = rootDir
        self.transform = PNGToMFCC()
        self.data = datasets.ImageFolder(
            root=self.rootDir,
            transform=self.transform
        )
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
    data = GTZANDataset(rootDir=r"..//dataset//archive//Data//images_original")
    data1 = data(train="True")
    data2 = data(train="False")

    print(len(data1))
    print(len(data2))
    print(data1[10])
    print(data1[1][0].shape)
