import librosa
import torch

import os
import numpy as np

# 指定音频文件所在的目录
directory = 'data/train'

# 获取目录下所有音频文件的路径
audio_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav')]

# 存储所有音频的MFCC特征向量的列表
all_mfccs_flat = []

# 循环处理每个音频文件
for audio_path in audio_paths:
    # 载入音频文件
    y, sr = librosa.load(audio_path)
    
    # 提取MFCC特征
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # 将MFCC特征转换为一维向量并添加到列表中
    mfccs_flat = mfccs.flatten()
    all_mfccs_flat.append(mfccs_flat)

# 将每个MFCC特征向量转换为PyTorch张量
all_mfccs_tensors = [torch.tensor(mfcc, dtype=torch.float32) for mfcc in all_mfccs_flat]
# print(all_mfccs_tensors)

# 使用pad_sequence函数填充序列到相同长度（如果需要的话）
# 首先需要确定最大长度
max_len = max([mfcc.size(0) for mfcc in all_mfccs_tensors])
# print(max_len) 1313

# 使用pad_sequence函数填充序列
from torch.nn.utils.rnn import pad_sequence

padded_mfccs_tensors = pad_sequence(all_mfccs_tensors, batch_first=True, padding_value=0)
# print(padded_mfccs_tensors)

# 假设你有一个标签列表，每个标签对应于 MFCC 序列
labels = torch.tensor([1, 2, 3])

from torch.utils.data import Dataset, DataLoader
class soundDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    # 根据给定的索引返回对应特征值/标签
    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]
        return feature, label
    
train_dataset = soundDataset(features=padded_mfccs_tensors, labels=labels)
print(f"特征值：{train_dataset.features}")
print(f"标签：{train_dataset.labels}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
for batch_features, batch_labels in train_loader:
    # 在这里对每个批次的数据进行操作
    print(batch_features)
    print(batch_labels)
    break
