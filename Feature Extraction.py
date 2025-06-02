import librosa
import pandas as pd # 导入Pandas用于数据处理
import os
import numpy as np
import librosa.display
import IPython.display as ipd
from scipy.io import wavfile
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from glob import glob # 用于文件路径匹配
from Data_Preparation_and_Preprocessing import df_main
import shutil # 用于文件操作，例如删除文件夹
import pickle

# 启用tqdm的pandas扩展
tqdm.pandas()

# 获取所有音频文件路径
audio_files_path = './Data/ICBHI_final_database'
audio_files = glob(os.path.join(audio_files_path, '**/*.wav'), recursive=True)

# 使用Librosa加载音频
y, sr = librosa.load(audio_files[0])
classes = list(df_main['Diagnosis'].unique()) # 获取独特的诊断类别

# 定义音频包络函数
def envelope(y, sr, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs) # 取音频信号的绝对值
    y_rolling = y.rolling(window=int(sr/10), min_periods=1, center=True).mean() # 计算移动平均值
    for i in y_rolling:
        if i > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask

# 清理后的音频文件夹路径
folder_path = './Data/wavfiles_cleaned'
if os.path.exists(folder_path):
    shutil.rmtree(folder_path)

os.makedirs(folder_path, exist_ok=True)

for i in tqdm(df_main['audio_file_name']):
    y, sr = librosa.load(audio_files_path + "/" + str(i), sr=22050) # 加载音频文件
    mask = envelope(y, sr, 0.0005) # 应用包络
    wavfile.write(filename='./Data/wavfiles_cleaned/' + str(i), rate=sr, data=y[mask]) # 保存清理后的音频

# 定义解析函数，用于提取音频特征
def parser(row):
    audio_file_name = os.path.join('./Data/wavfiles_cleaned', str(row['audio_file_name']))
    y, sr = librosa.load(audio_file_name, sr=22050) # 加载音频
    
    # 提取音频特征
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=24).T, axis=0) # MFCC 24个特征
    chromagrams = np.mean(librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=24).T, axis=0) # 色谱图  (24个特征)
    mel_specs = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0) # Mel频谱 128
    spec_contrasts = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6).T, axis=0) # 频谱对比  (7个特征)
    tonal_centroids = np.mean(librosa.feature.tonnetz(y=y, sr=sr).T, axis=0) # Tonnetz特征 (6个特征)
    
    # 将特征拼接成一个数组
    features = np.concatenate([mfccs, chromagrams, mel_specs, spec_contrasts, tonal_centroids], axis=0) #189个特征
    labels = row['Diagnosis'] # 提取诊断标签
    
    return [features, labels]


print("开始提取音频特征...")
data = df_main.progress_apply(parser, axis=1).tolist()

# 定义保存路径
folder_path = './Data'
os.makedirs(folder_path, exist_ok=True)  # 如果文件夹不存在，则创建它

# 保存data到文件
file_path = os.path.join(folder_path, 'data.pkl')  # 定义保存的文件名和路径

# 使用pickle保存数据
with open(file_path, 'wb') as f:
    pickle.dump(data, f)

print(f"数据已保存到 {file_path}")
print(f"总共处理了 {len(data)} 个音频文件")