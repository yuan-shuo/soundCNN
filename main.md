# 2024.4.9(特征向量提取)

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

   