import numpy as np
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

# 读取患者诊断数据
diagnosis_df = pd.read_csv('./Data/patient_diagnosis.csv', names=['Patient number', 'Diagnosis'])

# 绘制各诊断类别的数量统计图
plt.figure(figsize=(10,5))
sns.countplot(diagnosis_df, x='Diagnosis')
plt.title('Count of each Diagnosis')
plt.savefig("./Count of each Diagnosis.jpg")

# 读取患者人口统计信息
patient_df = pd.read_csv('./Data/demographic_info.txt', 
                         names=['Patient number', 'Age', 'Sex', 'Adult BMI (kg/m2)', 'Child Weight (kg)', 'Child Height (cm)'],
                         delimiter = ' ')

# 打印数据表头信息
print(patient_df.head())

# 检查BMI的缺失值数量
patient_df['Adult BMI (kg/m2)'].isna().sum()

# 根据儿童的身高和体重计算缺失的成人BMI值
for i in range(len(patient_df)):
    if (pd.isna(patient_df['Adult BMI (kg/m2)'][i])) and (not pd.isna(patient_df['Child Weight (kg)'][i])) and (not pd.isna(patient_df['Child Height (cm)'][i])):
        patient_df['Adult BMI (kg/m2)'][i] = round(patient_df['Child Weight (kg)'][i] / np.square(0.01 * patient_df['Child Height (cm)'][i]), 2)

# 再次检查BMI的缺失值数量
patient_df['Adult BMI (kg/m2)'].isna().sum()

# 合并诊断信息和人口统计信息
df = pd.merge(left=patient_df, right=diagnosis_df, how='left')
print(df.head())

# 解析音频文件名信息
Patient_numbers, Recording_indices, Chest_locations, Acquisition_modes, Recording_equipments = [], [], [], [], []
folder_path = './Data/ICBHI_final_database'

for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        parts = filename.split('_')
        Patient_number = parts[0]
        Recording_index = parts[1]
        Chest_location = parts[2]
        Acquisition_mode = parts[3]
        Recording_equipment = parts[4].split('.')[0]
        
        Patient_numbers.append(int(Patient_number))
        Recording_indices.append(Recording_index)
        Chest_locations.append(Chest_location)
        Acquisition_modes.append(Acquisition_mode)
        Recording_equipments.append(Recording_equipment)

df1 = pd.DataFrame({'Patient number': Patient_numbers, 
                    'Recording index': Recording_indices, 
                    'Chest location': Chest_locations,
                    'Acquisition mode': Acquisition_modes, 
                    'Recording equipment': Recording_equipments})

print(df1.tail())

df_all = pd.merge(left=df1, right=df, how='left').sort_values('Patient number').reset_index(drop=True)
print(df_all.head())

df_all['audio_file_name'] = df_all.apply(lambda row: 
                                         f"{row['Patient number']}_{row['Recording index']}_{row['Chest location']}_{row['Acquisition mode']}_{row['Recording equipment']}.wav", axis=1)

df_main = df_all[['Patient number', 'audio_file_name', 'Diagnosis']]
print(df_main.head(20))

# 绘制诊断类别数量统计图
plt.figure(figsize=(10,5))
sns.countplot(df_main, x='Diagnosis')
plt.title('Count of each Diagnosis in Main Dataframe')
plt.savefig("./Count of each Diagnosis in Main Dataframe.jpg")

# 移除不需要的诊断类别
df_main = df_main[(df_main['Diagnosis'] != 'Asthma') & (df_main['Diagnosis'] != 'LRTI')]
df_main = df_main.sort_values('Patient number').reset_index(drop=True)

# 检查诊断类别的分布
df_main['Diagnosis'].value_counts()
df_main['Diagnosis'].value_counts(normalize=True) * 100
