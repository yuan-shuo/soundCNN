# 2024.4.9(单个音频特征向量提取)

1. 用pip安装了librosa包，用来处理音频的

2. 通过代码得到输出

   ```python
   import librosa
   
   # 载入音频文件
   audio_path = 'data/1.wav'
   y, sr = librosa.load(audio_path)
   
   # 提取MFCC特征
   mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
   print(mfccs)
   
   # 输出
   [[-730.71576   -730.71576   -730.71576   ... -604.6304    -600.4629
     -582.9662   ]
    [   0.           0.           0.        ...   96.20656     94.39346
       88.19177  ]
    [   0.           0.           0.        ...   20.315022    20.19445
        4.4197907]
    ...
    [   0.           0.           0.        ...   -4.408634    -3.02205
        1.1847875]
    [   0.           0.           0.        ...   -6.3485336  -10.78275
      -19.176142 ]
    [   0.           0.           0.        ...   -6.1174374   -9.290737
      -16.357864 ]]
   ```

3. 理解mfccs的绘制

   ```
   me:所以我的mfccs内的列表每个都有82个数据，这就是我横轴的长度，然后每个x值对应13个y，对吗
   
   ai:是的，您理解得正确。如果您的 `mfccs` 变量是一个二维数组，其中每一列有82个数据，那么每一列对应一个时间点或帧，也就是说，横轴的长度将为82。
   
   而每个时间点或帧上，通常会有多个MFCC系数值。在大多数情况下，这些系数的数量通常是固定的，例如13个MFCC系数。因此，对于每个时间点或帧，您会有13个MFCC系数值。这些MFCC系数值构成了纵轴的数据。
   ```

4. 将mfccs二维转一维

   ```python
   # 将MFCC特征转换为一维向量
   mfccs_flat = mfccs.flatten() # (82, 13) => (82*13=1066)
   
   print("MFCC特征向量的长度:", len(mfccs_flat)) # MFCC特征向量的长度: 1066
   ```

5. 小结

   ```python
   import librosa
   
   # 载入音频文件
   audio_path = 'data/1.wav'
   y, sr = librosa.load(audio_path)
   
   # 提取MFCC特征
   mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
   # print(mfccs) => [[x1,x2,...,x82], [x1,x2,...,x82](内部共13(n_mfcc=13)个列表，每个列表长度为82)]
   # print(len(mfccs[0]))
   
   # 可视化MFCC特征
   # import matplotlib.pyplot as plt
   # plt.figure(figsize=(10, 4))
   # librosa.display.specshow(mfccs, x_axis='time')
   # plt.colorbar()
   # plt.title('MFCC')
   # plt.tight_layout()
   # plt.show()
   
   # 将MFCC特征转换为一维向量
   mfccs_flat = mfccs.flatten() # (82, 13) => (82*13=1066)
   
   print("MFCC特征向量的长度:", len(mfccs_flat)) # MFCC特征向量的长度: 1066
   ```

   

# 2024.4.10（数据集制作）

1. 直接看代码

   ```python
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
   
   # 输出
   特征值：tensor([[-730.7158, -730.7158, -730.7158,  ...,    0.0000,    0.0000,
               0.0000],
           [-636.5535, -636.5535, -636.5535,  ...,    0.0000,    0.0000,
               0.0000],
           [-665.5139, -665.5139, -665.5139,  ...,   -6.7150,  -11.0805,
             -12.6958]])
   标签：tensor([1, 2, 3])
   tensor([[-636.5535, -636.5535, -636.5535,  ...,    0.0000,    0.0000,
               0.0000],
           [-665.5139, -665.5139, -665.5139,  ...,   -6.7150,  -11.0805,
             -12.6958],
           [-730.7158, -730.7158, -730.7158,  ...,    0.0000,    0.0000,
               0.0000]])
   tensor([2, 3, 1])
   ```

   