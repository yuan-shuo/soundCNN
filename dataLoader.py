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