# 规划

## 制作流程

1. 将歌曲简化表示
2. 训练神经网络分类
3. 使用分类器进行分类实践

## 具体计划

[参考流程](https://kns.cnki.net/kcms2/article/abstract?v=tYqgYjzYKsDHYYYhGkmVHuF8gJYXLPEwrqqnfrPPytAzE2S0DdeW-0oBZX88xneqs8BjpieZwhvXHjSB6578B5WJU_nmGcbSnwLYtP89xkePRgenoBoiO3ByjBYQRDma7hBBZFiqCa30wUbBzBW7bOGmMCzvQDiO&uniplatform=NZKPT)：
（1）预处理。在音乐流派分类研究中，来自不同数据源的音乐文件可能具有不同的存储格式和采样频
率，且原始音乐信号中可能存在干扰噪声等无用信息。因此，预处理是音乐流派分类研究的第一步和基本
环节。主要包括统一信号格式、采样量化、成帧、加窗、预加重等，为后续的特征提取步骤提供标准化、可靠
的音乐样本。
（2）特征提取。特征提取是整个分类过程的关键部分，因为特征的选择和提取在很大程度上决定了分
类效果，主要目的是找出更贴合音乐信号特征的关键特征参数。在音乐流派分类领域，常用的特征参数包
括短时能量、过零率、短时谱、线性预测系数（LPC）和 Mel 倒谱系数。
（3）训练分类器。训练分类器是决定分类最终性能的关键模块。通过构建训练样本集对分类器进行训
练，找到合适的参数，最终得到训练好的分类器。
（4）分类结果测试。用于测试分类器的效果。待预测的新音乐样本进入分类器，利用训练好的分类器
对待预测的样本进行分类和预测，判断其是否属于该类别，最后统计分类准确率。

### 音频转换简化

将音乐的音频进行傅里叶变换转化为 Mel 频谱图和 MFCC

数据集：[GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)(比较小，可以用作前期的实验数据集)，  [MSD, tagtraum](https://www.tagtraum.com/msd_genre_datasets.html)(比较大，可用在最后完成时)
更多的数据集参见：[这篇文章](https://blog.csdn.net/linxid/article/details/87980916)

提取MFCC特征：librosa

可能的改进：[听觉图像引入音乐流派自动分类系统](https://kns.cnki.net/kcms2/article/abstract?v=tYqgYjzYKsDWzHVhvZGoijLYSEXTvLHlbywLZkxVWUkj1ZoXV1_W-0RpvI--tWlrABosbOQLNtEryVq3_uPLVUNz5t8zKyUZpcjGvlB6i-rCz57RDeMD2PaAnl_3yI17bLpiCJ9pVX75m9l4_6oVtgy7ZtIVUkyN&uniplatform=NZKPT)

### 分类器

卷积神经网络(CNN)
也许会使用Pytorch

### 参考资料

1. [python处理音频信号实战：手把手教你实现音乐流派分类和特征提取](https://zhuanlan.zhihu.com/p/54561504)(解释的比较详细，且有开源代码，但tensorflow)
2. [使用深度学习进行音频分类的端到端示例和解释](https://zhuanlan.zhihu.com/p/358241055)(同样详细且有代码)
3. [基于神经网络的音乐流派分类](https://zhuanlan.zhihu.com/p/54035086)(比较详细，在GitHub上有开源代码)
