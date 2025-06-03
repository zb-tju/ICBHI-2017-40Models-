import pickle
import numpy as np
from keras.models import Sequential, Model
from keras.layers import (Conv1D, Dense, Flatten, Dropout, BatchNormalization, 
                         MaxPooling1D, GlobalAveragePooling1D, LSTM, GRU, 
                         Bidirectional, Input, Add, Concatenate, Activation,
                         AveragePooling1D, SeparableConv1D, DepthwiseConv1D,
                         MultiHeadAttention, LayerNormalization, Embedding,
                         Lambda, Reshape, RepeatVector, TimeDistributed,
                         LocallyConnected1D, GlobalMaxPooling1D, Layer)
from keras.optimizers import Adam, SGD, RMSprop
from keras.regularizers import l1, l2, l1_l2
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import keras.backend as K
from tqdm import tqdm
import time
import tensorflow as tf
import os
import random

# =================== 设置随机种子 ===================
def set_seeds(seed=42):
    """设置所有随机种子以确保可重复性"""
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 可选：为了完全的确定性，但会降低性能
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'

# =================== 设置GPU ===================
def setup_gpu(gpu_id=0):
    """设置使用哪块GPU"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # 设置GPU内存增长
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # 如果指定了特定GPU
            if gpu_id is not None and gpu_id < len(gpus):
                tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
                print(f"使用GPU {gpu_id}: {gpus[gpu_id]}")
            else:
                print(f"使用所有可用GPU: {gpus}")
        except RuntimeError as e:
            print(f"GPU设置错误: {e}")
    else:
        print("没有检测到GPU，使用CPU")

# 设置随机种子
set_seeds(42)

# 设置使用GPU 0（可以修改为0,1,2,3中的任意一个）
setup_gpu(gpu_id=2)


class MambaBlock(Layer):
    """Mamba块的自定义实现"""
    def __init__(self, d_model=128, **kwargs):
        super(MambaBlock, self).__init__(**kwargs)
        self.d_model = d_model
        
    def build(self, input_shape):
        super(MambaBlock, self).build(input_shape)
        self.dense_proj = Dense(self.d_model * 2)
        self.dense_delta = Dense(self.d_model, activation='softplus')
        self.dense_A = Dense(self.d_model, activation='sigmoid')
        self.dense_B = Dense(self.d_model)
        self.dense_C = Dense(self.d_model)
        
    def call(self, inputs):
        x_proj = self.dense_proj(inputs)
        x = x_proj[..., :self.d_model]
        z = x_proj[..., self.d_model:]
        
        delta = self.dense_delta(x)
        A = self.dense_A(x)
        B = self.dense_B(x)
        C = self.dense_C(x)
        
        # 简化的SSM计算
        h = x * A + B
        y = C * h
        
        # 门控
        y = y * K.sigmoid(z)
        return y

class LinearAttentionLayer(Layer):
    """线性注意力层的自定义实现"""
    def __init__(self, units=128, **kwargs):
        super(LinearAttentionLayer, self).__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        super(LinearAttentionLayer, self).build(input_shape)
        self.query_dense = Dense(self.units)
        self.key_dense = Dense(self.units)
        self.value_dense = Dense(self.units)
        
    def call(self, inputs):
        q = self.query_dense(inputs)
        k = self.key_dense(inputs)
        v = self.value_dense(inputs)
        
        # 线性化
        q = K.exp(q)
        k = K.exp(k)
        
        # 计算注意力
        kv = K.batch_dot(k, v, axes=[1, 1])
        qkv = K.batch_dot(q, kv, axes=[2, 1])
        
        return qkv

class PositionalEncoding(Layer):
    """位置编码层"""
    def __init__(self, num_patches, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.d_model = d_model
        
    def build(self, input_shape):
        super(PositionalEncoding, self).build(input_shape)
        # 创建可学习的位置编码
        self.pos_embed = self.add_weight(
            name='pos_embed',
            shape=(1, self.num_patches, self.d_model),
            initializer='zeros',
            trainable=True
        )
        
    def call(self, inputs):
        return inputs + self.pos_embed
# =================== 数据加载和预处理 ===================
file_path = './Data/data.pkl'
with open(file_path, 'rb') as f:
    data = pickle.load(f)
    
X = np.array(list(zip(*data))[0])
y = np.array(list(zip(*data))[1])

sampling_strategy_over = {'Pneumonia': 150, 'Healthy': 150, 'URTI': 100, 'Bronchiectasis': 100, 'Bronchiolitis': 100}
smote = SMOTE(sampling_strategy=sampling_strategy_over, k_neighbors=5, random_state=42)
X_resampled_over, y_resampled_over = smote.fit_resample(X, y)

copd_indices = np.where(y_resampled_over == 'COPD')[0]
np.random.seed(42)
np.random.shuffle(copd_indices)
remove_indices = copd_indices[:len(copd_indices) // 2]

X_resampled = np.delete(X_resampled_over, remove_indices, axis=0)
y_resampled = np.delete(y_resampled_over, remove_indices)

print('Shape of X is', X_resampled.shape)
print('Shape of y is', y_resampled.shape)

le = LabelEncoder()
y_encoded = le.fit_transform(y_resampled)
y_resampled = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
X_train_reshaped = np.reshape(X_train, (X_train.shape[0], 189, 1))
X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 189, 1))

input_shape = (189, 1)


class Model1_BasicCNN:
    """基础CNN模型"""
    def __init__(self, input_shape=(189, 1)):
        self.input_shape = input_shape
        
    def build(self):
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=self.input_shape),
            Dropout(0.5),
            Flatten(),
            Dense(6, activation='softmax')
        ])
        return model

class Model2_DeepCNN:
    """深层CNN模型"""
    def __init__(self, input_shape=(189, 1)):
        self.input_shape = input_shape
        
    def build(self):
        model = Sequential([
            Conv1D(32, 3, activation='relu', input_shape=self.input_shape),
            Conv1D(64, 3, activation='relu'),
            Conv1D(128, 3, activation='relu'),
            MaxPooling1D(2),
            Dropout(0.3),
            Conv1D(256, 3, activation='relu'),
            GlobalAveragePooling1D(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(6, activation='softmax')
        ])
        return model

class Model3_BatchNormCNN:
    """批标准化CNN模型"""
    def __init__(self, input_shape=(189, 1)):
        self.input_shape = input_shape
        
    def build(self):
        model = Sequential([
            Conv1D(64, 5, activation='relu', input_shape=self.input_shape),
            BatchNormalization(),
            Conv1D(128, 3, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(2),
            Conv1D(256, 3, activation='relu'),
            BatchNormalization(),
            GlobalAveragePooling1D(),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(6, activation='softmax')
        ])
        return model

class Model4_LSTM:
    """LSTM模型"""
    def __init__(self, input_shape=(189, 1)):
        self.input_shape = input_shape
        
    def build(self):
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=self.input_shape),
            Dropout(0.3),
            LSTM(64, return_sequences=False),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(6, activation='softmax')
        ])
        return model

class Model5_BiLSTM:
    """双向LSTM模型"""
    def __init__(self, input_shape=(189, 1)):
        self.input_shape = input_shape
        
    def build(self):
        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True), input_shape=self.input_shape),
            Dropout(0.3),
            Bidirectional(LSTM(32)),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(6, activation='softmax')
        ])
        return model

class Model6_GRU:
    """GRU模型"""
    def __init__(self, input_shape=(189, 1)):
        self.input_shape = input_shape
        
    def build(self):
        model = Sequential([
            GRU(128, return_sequences=True, input_shape=self.input_shape),
            Dropout(0.3),
            GRU(64),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(6, activation='softmax')
        ])
        return model

class Model7_CNN_LSTM:
    """CNN+LSTM混合模型"""
    def __init__(self, input_shape=(189, 1)):
        self.input_shape = input_shape
        
    def build(self):
        model = Sequential([
            Conv1D(64, 3, activation='relu', input_shape=self.input_shape),
            Conv1D(128, 3, activation='relu'),
            MaxPooling1D(2),
            LSTM(64, return_sequences=True),
            LSTM(32),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(6, activation='softmax')
        ])
        return model

class Model8_CNN_GRU:
    """CNN+GRU混合模型"""
    def __init__(self, input_shape=(189, 1)):
        self.input_shape = input_shape
        
    def build(self):
        model = Sequential([
            Conv1D(64, 5, activation='relu', input_shape=self.input_shape),
            BatchNormalization(),
            Conv1D(128, 3, activation='relu'),
            MaxPooling1D(2),
            GRU(64, return_sequences=True),
            GRU(32),
            Dense(64, activation='relu'),
            Dropout(0.4),
            Dense(6, activation='softmax')
        ])
        return model

class Model9_ResidualCNN:
    """残差CNN模型"""
    def __init__(self, input_shape=(189, 1)):
        self.input_shape = input_shape
        
    def build(self):
        inputs = Input(shape=self.input_shape)
        
        # 第一层
        x = Conv1D(64, 3, padding='same', activation='relu')(inputs)
        
        # 残差块
        shortcut = Conv1D(64, 1, padding='same')(x)
        x = Conv1D(64, 3, padding='same', activation='relu')(x)
        x = Conv1D(64, 3, padding='same')(x)
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        
        x = MaxPooling1D(2)(x)
        x = Conv1D(128, 3, activation='relu')(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(6, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        return model

class Model10_MultiScale:
    """多尺度CNN模型"""
    def __init__(self, input_shape=(189, 1)):
        self.input_shape = input_shape
        
    def build(self):
        inputs = Input(shape=self.input_shape)
        
        # 多尺度卷积
        conv1 = Conv1D(32, 3, activation='relu', padding='same')(inputs)
        conv2 = Conv1D(32, 5, activation='relu', padding='same')(inputs)
        conv3 = Conv1D(32, 7, activation='relu', padding='same')(inputs)
        
        # 合并
        merged = Concatenate()([conv1, conv2, conv3])
        x = BatchNormalization()(merged)
        x = MaxPooling1D(2)(x)
        x = Conv1D(128, 3, activation='relu')(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(6, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        return model

class Model11_DenseCNN:
    """密集连接CNN模型"""
    def __init__(self, input_shape=(189, 1)):
        self.input_shape = input_shape
        
    def build(self):
        model = Sequential([
            Conv1D(64, 3, activation='relu', input_shape=self.input_shape),
            Conv1D(128, 3, activation='relu'),
            Conv1D(256, 3, activation='relu'),
            MaxPooling1D(2),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(6, activation='softmax')
        ])
        return model

class Model12_SeparableCNN:
    """可分离卷积模型"""
    def __init__(self, input_shape=(189, 1)):
        self.input_shape = input_shape
        
    def build(self):
        model = Sequential([
            SeparableConv1D(64, 3, activation='relu', input_shape=self.input_shape),
            BatchNormalization(),
            SeparableConv1D(128, 3, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(2),
            SeparableConv1D(256, 3, activation='relu'),
            GlobalAveragePooling1D(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(6, activation='softmax')
        ])
        return model

class Model13_AttentionLSTM:
    """注意力LSTM模型"""
    def __init__(self, input_shape=(189, 1)):
        self.input_shape = input_shape
        
    def build(self):
        inputs = Input(shape=self.input_shape)
        
        # LSTM层
        lstm_out = LSTM(128, return_sequences=True)(inputs)
        
        # 简单注意力机制
        attention = Dense(1, activation='tanh')(lstm_out)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(128)(attention)
        attention = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(attention)
        
        # 应用注意力
        merged = Lambda(lambda x: x[0] * x[1])([lstm_out, attention])
        merged = Lambda(lambda x: K.sum(x, axis=1))(merged)
        
        output = Dense(128, activation='relu')(merged)
        output = Dropout(0.5)(output)
        output = Dense(6, activation='softmax')(output)
        
        model = Model(inputs, output)
        return model

class Model14_WideDeepCNN:
    """宽深CNN模型"""
    def __init__(self, input_shape=(189, 1)):
        self.input_shape = input_shape
        
    def build(self):
        inputs = Input(shape=self.input_shape)
        
        # 宽部分
        wide = Conv1D(256, 1, activation='relu')(inputs)
        wide = GlobalAveragePooling1D()(wide)
        
        # 深部分
        deep = Conv1D(64, 3, activation='relu')(inputs)
        deep = Conv1D(128, 3, activation='relu')(deep)
        deep = MaxPooling1D(2)(deep)
        deep = Conv1D(256, 3, activation='relu')(deep)
        deep = GlobalAveragePooling1D()(deep)
        
        # 合并
        merged = Concatenate()([wide, deep])
        output = Dense(128, activation='relu')(merged)
        output = Dropout(0.5)(output)
        output = Dense(6, activation='softmax')(output)
        
        model = Model(inputs, output)
        return model

class Model15_PyramidCNN:
    """金字塔CNN模型"""
    def __init__(self, input_shape=(189, 1)):
        self.input_shape = input_shape
        
    def build(self):
        model = Sequential([
            Conv1D(32, 7, activation='relu', input_shape=self.input_shape),
            MaxPooling1D(2),
            Conv1D(64, 5, activation='relu'),
            MaxPooling1D(2),
            Conv1D(128, 3, activation='relu'),
            MaxPooling1D(2),
            Conv1D(256, 3, activation='relu'),
            GlobalAveragePooling1D(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(6, activation='softmax')
        ])
        return model

class Model16_StackedLSTM:
    """堆叠LSTM模型"""
    def __init__(self, input_shape=(189, 1)):
        self.input_shape = input_shape
        
    def build(self):
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=self.input_shape),
            Dropout(0.3),
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(6, activation='softmax')
        ])
        return model

class Model17_DilatedCNN:
    """扩张卷积模型"""
    def __init__(self, input_shape=(189, 1)):
        self.input_shape = input_shape
        
    def build(self):
        model = Sequential([
            Conv1D(64, 3, dilation_rate=1, activation='relu', padding='same', input_shape=self.input_shape),
            Conv1D(64, 3, dilation_rate=2, activation='relu', padding='same'),
            Conv1D(128, 3, dilation_rate=4, activation='relu', padding='same'),
            Conv1D(128, 3, dilation_rate=8, activation='relu', padding='same'),
            GlobalAveragePooling1D(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(6, activation='softmax')
        ])
        return model

class Model18_LocallyConnected:
    """局部连接模型"""
    def __init__(self, input_shape=(189, 1)):
        self.input_shape = input_shape
        
    def build(self):
        model = Sequential([
            LocallyConnected1D(64, 3, activation='relu', input_shape=self.input_shape),
            Dropout(0.3),
            LocallyConnected1D(128, 3, activation='relu'),
            Dropout(0.3),
            GlobalAveragePooling1D(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(6, activation='softmax')
        ])
        return model

class Model19_ELU_CNN:
    """ELU激活函数CNN模型"""
    def __init__(self, input_shape=(189, 1)):
        self.input_shape = input_shape
        
    def build(self):
        model = Sequential([
            Conv1D(64, 3, activation='elu', input_shape=self.input_shape),
            BatchNormalization(),
            Conv1D(128, 3, activation='elu'),
            BatchNormalization(),
            MaxPooling1D(2),
            Conv1D(256, 3, activation='elu'),
            BatchNormalization(),
            GlobalAveragePooling1D(),
            Dense(128, activation='elu'),
            Dropout(0.5),
            Dense(6, activation='softmax')
        ])
        return model

class Model20_Swish_CNN:
    """Swish激活函数CNN模型"""
    def __init__(self, input_shape=(189, 1)):
        self.input_shape = input_shape
        
    def build(self):
        def swish(x):
            return x * K.sigmoid(x)
        
        model = Sequential([
            Conv1D(64, 3, input_shape=self.input_shape),
            Lambda(swish),
            BatchNormalization(),
            Conv1D(128, 3),
            Lambda(swish),
            BatchNormalization(),
            MaxPooling1D(2),
            Conv1D(256, 3),
            Lambda(swish),
            GlobalAveragePooling1D(),
            Dense(128),
            Lambda(swish),
            Dropout(0.5),
            Dense(6, activation='softmax')
        ])
        return model

class Model21_L1Regularized:
    """L1正则化模型"""
    def __init__(self, input_shape=(189, 1)):
        self.input_shape = input_shape
        
    def build(self):
        model = Sequential([
            Conv1D(64, 3, activation='relu', kernel_regularizer=l1(0.01), input_shape=self.input_shape),
            Conv1D(128, 3, activation='relu', kernel_regularizer=l1(0.01)),
            MaxPooling1D(2),
            Conv1D(256, 3, activation='relu', kernel_regularizer=l1(0.01)),
            GlobalAveragePooling1D(),
            Dense(128, activation='relu', kernel_regularizer=l1(0.01)),
            Dropout(0.5),
            Dense(6, activation='softmax')
        ])
        return model

class Model22_L2Regularized:
    """L2正则化模型"""
    def __init__(self, input_shape=(189, 1)):
        self.input_shape = input_shape
        
    def build(self):
        model = Sequential([
            Conv1D(64, 3, activation='relu', kernel_regularizer=l2(0.01), input_shape=self.input_shape),
            Conv1D(128, 3, activation='relu', kernel_regularizer=l2(0.01)),
            MaxPooling1D(2),
            Conv1D(256, 3, activation='relu', kernel_regularizer=l2(0.01)),
            GlobalAveragePooling1D(),
            Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            Dense(6, activation='softmax')
        ])
        return model

class Model23_DoubleConv:
    """双卷积模型"""
    def __init__(self, input_shape=(189, 1)):
        self.input_shape = input_shape
        
    def build(self):
        model = Sequential([
            Conv1D(32, 3, activation='relu', input_shape=self.input_shape),
            Conv1D(32, 3, activation='relu'),
            MaxPooling1D(2),
            Conv1D(64, 3, activation='relu'),
            Conv1D(64, 3, activation='relu'),
            MaxPooling1D(2),
            Conv1D(128, 3, activation='relu'),
            Conv1D(128, 3, activation='relu'),
            GlobalAveragePooling1D(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(6, activation='softmax')
        ])
        return model

class Model24_GlobalMaxPool:
    """全局最大池化模型"""
    def __init__(self, input_shape=(189, 1)):
        self.input_shape = input_shape
        
    def build(self):
        model = Sequential([
            Conv1D(64, 3, activation='relu', input_shape=self.input_shape),
            Conv1D(128, 3, activation='relu'),
            Conv1D(256, 3, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(6, activation='softmax')
        ])
        return model

class Model25_AvgPool:
    """平均池化模型"""
    def __init__(self, input_shape=(189, 1)):
        self.input_shape = input_shape
        
    def build(self):
        model = Sequential([
            Conv1D(64, 3, activation='relu', input_shape=self.input_shape),
            AveragePooling1D(2),
            Conv1D(128, 3, activation='relu'),
            AveragePooling1D(2),
            Conv1D(256, 3, activation='relu'),
            GlobalAveragePooling1D(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(6, activation='softmax')
        ])
        return model

class Model26_MambaInspired:
    """Mamba启发的选择性状态空间模型"""
    def __init__(self, input_shape=(189, 1)):
        self.input_shape = input_shape
        
    def build(self):
        inputs = Input(shape=self.input_shape)
        
        # 特征嵌入
        x = Conv1D(128, 1, activation='linear')(inputs)
        
        # 选择性门控机制 (简化版Mamba核心思想)
        # 输入门：决定哪些信息需要更新
        input_gate = Dense(128, activation='sigmoid')(x)
        
        # 遗忘门：决定哪些信息需要遗忘  
        forget_gate = Dense(128, activation='sigmoid')(x)
        
        # 候选状态
        candidate = Dense(128, activation='tanh')(x)
        
        # 状态更新 (类似LSTM但更简化)
        def selective_update(inputs):
            x, i_gate, f_gate, candidate = inputs
            # 模拟状态空间更新
            state = f_gate * x + i_gate * candidate
            return state
        
        state = Lambda(selective_update)([x, input_gate, forget_gate, candidate])
        
        # 输出门
        output_gate = Dense(128, activation='sigmoid')(state)
        output = Lambda(lambda x: x[0] * K.tanh(x[1]))([output_gate, state])
        
        # 时序建模
        lstm_out = LSTM(64, return_sequences=True)(output)
        lstm_out = LSTM(32)(lstm_out)
        
        # 分类器
        dense = Dense(128, activation='relu')(lstm_out)
        dropout = Dropout(0.5)(dense)
        outputs = Dense(6, activation='softmax')(dropout)
        
        model = Model(inputs, outputs)
        return model

class Model27_TransformerLike:
    """Transformer风格模型"""
    def __init__(self, input_shape=(189, 1)):
        self.input_shape = input_shape
        
    def build(self):
        inputs = Input(shape=self.input_shape)
        
        # 位置编码
        x = Conv1D(128, 1)(inputs)
        
        # 多头注意力
        attention = MultiHeadAttention(num_heads=8, key_dim=16)(x, x)
        attention = Dropout(0.1)(attention)
        
        # 残差连接
        x = Add()([x, attention])
        x = LayerNormalization()(x)
        
        # 前馈网络
        ff = Dense(256, activation='relu')(x)
        ff = Dropout(0.1)(ff)
        ff = Dense(128)(ff)
        
        # 再次残差连接
        x = Add()([x, ff])
        x = LayerNormalization()(x)
        
        # 全局池化
        x = GlobalAveragePooling1D()(x)
        
        # 分类器
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(6, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        return model



class Model28_PureMamba_Fixed:
    """修复后的纯Mamba状态空间模型"""
    def __init__(self, input_shape=(189, 1)):
        self.input_shape = input_shape
        
    def build(self):
        inputs = Input(shape=self.input_shape)
        
        # 输入投影
        x = Dense(128)(inputs)
        
        # 使用自定义Mamba块
        x = MambaBlock(d_model=128)(x)
        x = Dropout(0.2)(x)
        x = MambaBlock(d_model=128)(x)
        x = Dropout(0.2)(x)
        
        # 输出层
        x = GlobalAveragePooling1D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(6, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        return model

class Model29_MambaTransformer:
    """Mamba+Transformer混合模型"""
    def __init__(self, input_shape=(189, 1)):
        self.input_shape = input_shape
        
    def build(self):
        inputs = Input(shape=self.input_shape)
        
        # Mamba分支
        mamba_x = Dense(64)(inputs)
        mamba_gate = Dense(64, activation='sigmoid')(mamba_x)
        mamba_out = Lambda(lambda t: t[0] * t[1])([mamba_x, mamba_gate])
        
        # Transformer分支
        trans_x = Dense(64)(inputs)
        attention = MultiHeadAttention(num_heads=4, key_dim=16)(trans_x, trans_x)
        trans_out = Add()([trans_x, attention])
        trans_out = LayerNormalization()(trans_out)
        
        # 融合
        merged = Concatenate()([mamba_out, trans_out])
        x = Dense(128, activation='relu')(merged)
        x = GlobalAveragePooling1D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(6, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        return model

class Model30_ViTLike_Fixed:
    """修复后的Vision Transformer风格1D模型"""
    def __init__(self, input_shape=(189, 1)):
        self.input_shape = input_shape
        
    def build(self):
        inputs = Input(shape=self.input_shape)
        
        # Patch embedding - 调整patch大小确保整除
        patch_size = 9  # 189可以被9整除
        num_patches = self.input_shape[0] // patch_size  # 21
        
        # 重塑为patches
        x = Lambda(lambda t: K.reshape(t, (-1, num_patches, patch_size)))(inputs)
        x = Dense(128)(x)  # Patch embedding
        
        # 位置编码 - 使用自定义层（这里是关键修改）
        x = PositionalEncoding(num_patches=num_patches, d_model=128)(x)
        
        # Transformer层
        for _ in range(3):
            # 保存输入用于残差连接
            shortcut = x
            
            # 多头注意力
            attention = MultiHeadAttention(num_heads=8, key_dim=16)(x, x)
            x = Add()([shortcut, attention])
            x = LayerNormalization()(x)
            
            # MLP
            shortcut = x
            mlp = Dense(256, activation='gelu')(x)
            mlp = Dense(128)(mlp)
            x = Add()([shortcut, mlp])
            x = LayerNormalization()(x)
        
        # 分类头
        x = GlobalAveragePooling1D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(6, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        return model

class Model31_BertLike:
    """BERT风格编码器模型"""
    def __init__(self, input_shape=(189, 1)):
        self.input_shape = input_shape
        
    def build(self):
        inputs = Input(shape=self.input_shape)
        
        # 输入嵌入
        x = Dense(256)(inputs)
        
        # BERT编码器层
        for i in range(4):
            # 多头自注意力
            attention = MultiHeadAttention(
                num_heads=8, 
                key_dim=32,
                dropout=0.1
            )(x, x)
            attention = Dropout(0.1)(attention)
            
            # 残差连接和层归一化
            x = Add()([x, attention])
            x = LayerNormalization()(x)
            
            # 前馈网络
            ff = Dense(512, activation='gelu')(x)
            ff = Dropout(0.1)(ff)
            ff = Dense(256)(ff)
            
            # 残差连接和层归一化
            x = Add()([x, ff])
            x = LayerNormalization()(x)
        
        # 池化和分类
        x = GlobalAveragePooling1D()(x)
        x = Dense(256, activation='gelu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(6, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        return model

class Model32_GPTLike:
    """GPT风格解码器模型"""
    def __init__(self, input_shape=(189, 1)):
        self.input_shape = input_shape
        
    def build(self):
        inputs = Input(shape=self.input_shape)
        
        # 输入嵌入
        x = Dense(256)(inputs)
        
        # GPT解码器层（因果注意力）
        for i in range(4):
            # 因果注意力（简化版）
            attention = MultiHeadAttention(
                num_heads=8,
                key_dim=32,
                dropout=0.1
            )(x, x, use_causal_mask=True)
            attention = Dropout(0.1)(attention)
            
            # 残差连接
            x = Add()([x, attention])
            x = LayerNormalization()(x)
            
            # MLP
            mlp = Dense(1024, activation='gelu')(x)
            mlp = Dropout(0.1)(mlp)
            mlp = Dense(256)(mlp)
            
            x = Add()([x, mlp])
            x = LayerNormalization()(x)
        
        # 最后一个位置的输出
        x = Lambda(lambda t: t[:, -1, :])(x)
        x = Dense(256, activation='gelu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(6, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        return model

class Model33_LinearAttention_Fixed:
    """修复后的线性注意力模型"""
    def __init__(self, input_shape=(189, 1)):
        self.input_shape = input_shape
        
    def build(self):
        inputs = Input(shape=self.input_shape)
        
        # 特征映射
        x = Dense(128)(inputs)
        
        # 使用自定义线性注意力层
        x = LinearAttentionLayer(units=128)(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(6, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        return model

class Model34_SparseAttention:
    """稀疏注意力模型"""
    def __init__(self, input_shape=(189, 1)):
        self.input_shape = input_shape
        
    def build(self):
        inputs = Input(shape=self.input_shape)
        
        # 输入投影
        x = Dense(128)(inputs)
        
        # 稀疏注意力（局部窗口）
        window_size = 16
        for i in range(3):
            # 局部自注意力
            attention = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
            x = Add()([x, attention])
            x = LayerNormalization()(x)
            
            # 前馈
            ff = Dense(256, activation='relu')(x)
            ff = Dense(128)(ff)
            x = Add()([x, ff])
            x = LayerNormalization()(x)
        
        x = GlobalAveragePooling1D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(6, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        return model

class Model35_CrossAttention_Fixed:
    """修复后的交叉注意力模型"""
    def __init__(self, input_shape=(189, 1)):
        self.input_shape = input_shape
        
    def build(self):
        inputs = Input(shape=self.input_shape)
        
        # 两个并行分支，使用padding='same'保持序列长度一致
        branch1 = Conv1D(64, 3, activation='relu', padding='same')(inputs)
        branch2 = Conv1D(64, 5, activation='relu', padding='same')(inputs)
        
        # 交叉注意力
        cross_attn1 = MultiHeadAttention(num_heads=4, key_dim=16)(branch1, branch2)
        cross_attn2 = MultiHeadAttention(num_heads=4, key_dim=16)(branch2, branch1)
        
        # 融合
        merged = Concatenate()([cross_attn1, cross_attn2])
        x = Dense(128, activation='relu')(merged)
        x = GlobalAveragePooling1D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(6, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        return model

class Model36_SelfAttention:
    """纯自注意力模型"""
    def __init__(self, input_shape=(189, 1)):
        self.input_shape = input_shape
        
    def build(self):
        inputs = Input(shape=self.input_shape)
        
        # 多层自注意力
        x = Dense(128)(inputs)
        
        for i in range(6):
            attention = MultiHeadAttention(
                num_heads=8,
                key_dim=16,
                dropout=0.1
            )(x, x)
            x = Add()([x, attention])
            x = LayerNormalization()(x)
            x = Dropout(0.1)(x)
        
        x = GlobalAveragePooling1D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(6, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        return model

class Model37_HierarchicalTransformer:
    """层次化Transformer模型"""
    def __init__(self, input_shape=(189, 1)):
        self.input_shape = input_shape
        
    def build(self):
        inputs = Input(shape=self.input_shape)
        
        # 第一层：局部特征
        x = Dense(64)(inputs)
        local_attn = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
        x = Add()([x, local_attn])
        x = LayerNormalization()(x)
        
        # 下采样
        x = MaxPooling1D(2)(x)
        
        # 第二层：中层特征
        x = Dense(128)(x)
        mid_attn = MultiHeadAttention(num_heads=8, key_dim=16)(x, x)
        x = Add()([x, mid_attn])
        x = LayerNormalization()(x)
        
        # 再次下采样
        x = MaxPooling1D(2)(x)
        
        # 第三层：全局特征
        x = Dense(256)(x)
        global_attn = MultiHeadAttention(num_heads=8, key_dim=32)(x, x)
        x = Add()([x, global_attn])
        x = LayerNormalization()(x)
        
        x = GlobalAveragePooling1D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(6, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        return model

class Model38_ConvTransformer:
    """卷积Transformer模型"""
    def __init__(self, input_shape=(189, 1)):
        self.input_shape = input_shape
        
    def build(self):
        inputs = Input(shape=self.input_shape)
        
        # 卷积特征提取
        x = Conv1D(64, 7, activation='relu', padding='same')(inputs)
        x = Conv1D(128, 5, activation='relu', padding='same')(x)
        x = MaxPooling1D(2)(x)
        
        # Transformer层
        for i in range(3):
            attention = MultiHeadAttention(num_heads=8, key_dim=16)(x, x)
            x = Add()([x, attention])
            x = LayerNormalization()(x)
            
            # 卷积前馈网络
            ff = Conv1D(256, 1, activation='relu')(x)
            ff = Conv1D(128, 1)(ff)
            x = Add()([x, ff])
            x = LayerNormalization()(x)
        
        x = GlobalAveragePooling1D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(6, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        return model

class Model39_MegaLSTM:
    """超大LSTM模型"""
    def __init__(self, input_shape=(189, 1)):
        self.input_shape = input_shape
        
    def build(self):
        model = Sequential([
            Bidirectional(LSTM(256, return_sequences=True), input_shape=self.input_shape),
            Dropout(0.3),
            Bidirectional(LSTM(256, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(128, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(128, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(64)),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(6, activation='softmax')
        ])
        return model

class Model40_HybridMamba_Fixed:
    """修复后的混合Mamba架构"""
    def __init__(self, input_shape=(189, 1)):
        self.input_shape = input_shape
        
    def build(self):
        inputs = Input(shape=self.input_shape)
        
        # CNN分支，使用padding='same'保持序列长度
        cnn_branch = Conv1D(64, 3, activation='relu', padding='same')(inputs)
        cnn_branch = Conv1D(128, 3, activation='relu', padding='same')(cnn_branch)
        
        # Mamba分支
        mamba_x = Dense(128)(inputs)
        mamba_gate = Dense(128, activation='sigmoid')(mamba_x)
        mamba_forget = Dense(128, activation='sigmoid')(mamba_x)
        mamba_candidate = Dense(128, activation='tanh')(mamba_x)
        
        # Mamba状态更新
        mamba_state = Lambda(lambda t: t[0] * t[1] + t[2] * t[3])([
            mamba_forget, mamba_x, mamba_gate, mamba_candidate
        ])
        
        # LSTM分支
        lstm_branch = LSTM(128, return_sequences=True)(inputs)
        
        # 确保所有分支形状一致
        # cnn_branch: (None, 189, 128)
        # mamba_state: (None, 189, 128)
        # lstm_branch: (None, 189, 128)
        
        # 注意力融合
        concat_features = Concatenate()([cnn_branch, mamba_state, lstm_branch])
        attention_weights = Dense(1, activation='softmax')(concat_features)
        
        # 加权融合
        weighted_cnn = Lambda(lambda t: t[0] * t[1][:, :, 0:1])([cnn_branch, attention_weights])
        weighted_mamba = Lambda(lambda t: t[0] * t[1][:, :, 0:1])([mamba_state, attention_weights])
        weighted_lstm = Lambda(lambda t: t[0] * t[1][:, :, 0:1])([lstm_branch, attention_weights])
        
        # 最终融合
        final_output = Add()([weighted_cnn, weighted_mamba, weighted_lstm])
        x = GlobalAveragePooling1D()(final_output)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(6, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        return model

# 创建所有模型实例（包含修复后的模型）
models = [
    Model1_BasicCNN(), Model2_DeepCNN(), Model3_BatchNormCNN(), Model4_LSTM(),
    Model5_BiLSTM(), Model6_GRU(), Model7_CNN_LSTM(), Model8_CNN_GRU(),
    Model9_ResidualCNN(), Model10_MultiScale(), Model11_DenseCNN(), Model12_SeparableCNN(),
    Model13_AttentionLSTM(), Model14_WideDeepCNN(), Model15_PyramidCNN(), Model16_StackedLSTM(),
    Model17_DilatedCNN(), Model18_LocallyConnected(), Model19_ELU_CNN(), Model20_Swish_CNN(),
    Model21_L1Regularized(), Model22_L2Regularized(), Model23_DoubleConv(), Model24_GlobalMaxPool(),
    Model25_AvgPool(), Model26_MambaInspired(), Model27_TransformerLike(), 
    Model28_PureMamba_Fixed(),  
    Model29_MambaTransformer(), 
    Model30_ViTLike_Fixed(), 
    Model31_BertLike(), Model32_GPTLike(),
    Model33_LinearAttention_Fixed(), 
    Model34_SparseAttention(), 
    Model35_CrossAttention_Fixed(),  
    Model36_SelfAttention(),
    Model37_HierarchicalTransformer(), Model38_ConvTransformer(), Model39_MegaLSTM(), 
    Model40_HybridMamba_Fixed() 
]

# 训练并保存所有模型
model_names = [
    # 'basic_cnn', 'deep_cnn', 'batch_norm_cnn', 'lstm', 'bilstm', 'gru', 
    # 'cnn_lstm', 'cnn_gru', 'residual_cnn', 'multiscale', 'dense_cnn', 
    # 'separable_cnn', 'attention_lstm', 'wide_deep_cnn', 'pyramid_cnn', 
    # 'stacked_lstm', 'dilated_cnn', 'locally_connected', 'elu_cnn', 
    # 'swish_cnn', 'l1_regularized', 'l2_regularized', 'double_conv', 
    # 'global_max_pool', 'avg_pool', 'mamba_inspired', 'transformer_like',
    # 'pure_mamba', 'mamba_transformer', 'vit_like', 'bert_like', 'gpt_like',
    # 'linear_attention', 'sparse_attention', 'cross_attention', 'self_attention',
    # 'hierarchical_transformer', 'conv_transformer', 'mega_lstm', 'hybrid_mamba'
    'vit_like'
]

# 自定义进度条回调
class CustomTqdmCallback(Callback):
    def __init__(self, model_name, total_epochs):
        super().__init__()
        self.model_name = model_name
        self.total_epochs = total_epochs
        self.pbar = None
        
    def on_train_begin(self, logs=None):
        self.pbar = tqdm(total=self.total_epochs, desc=f"🔥 训练 {self.model_name}", 
                        unit="epoch", ncols=150, leave=False, 
                        bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}')
        
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            self.pbar.set_postfix({
                '📉 loss': f"{logs.get('loss', 0):.4f}",
                '📈 acc': f"{logs.get('accuracy', 0):.4f}",
                '📉 val_loss': f"{logs.get('val_loss', 0):.4f}",
                '📈 val_acc': f"{logs.get('val_accuracy', 0):.4f}"
            })
        self.pbar.update(1)
        
    def on_train_end(self, logs=None):
        if self.pbar:
            self.pbar.close()

print("🚀 开始训练所有模型...")
print("=" * 60)

# 记录训练结果
training_results = []
total_models = len(models)

# 总体进度条
overall_pbar = tqdm(total=total_models, desc="🚀 总体训练进度", position=0, leave=True, 
                   ncols=120, bar_format='{l_bar}{bar:40}{r_bar}{bar:-40b}',
                   colour='green')

for i, (model_class, model_name) in enumerate(zip(models, model_names)):
    start_time = time.time()
    
    print(f"\n" + "="*80)
    print(f"🔥 正在训练模型 {i+1}/{total_models}: {model_name.upper()}")
    print("="*80)
    
    try:
        # 构建模型
        model = model_class.build()
        
        # 编译模型
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # 显示模型参数数量
        total_params = model.count_params()
        print(f"📊 模型参数数量: {total_params:,}")
        print(f"⏰ 开始时间: {time.strftime('%H:%M:%S')}")
        print("-" * 80)
        
        # 创建进度条回调
        progress_callback = CustomTqdmCallback(model_name, 125)
        
        # 训练模型
        history = model.fit(
            X_train_reshaped, y_train, 
            batch_size=32, 
            epochs=125, 
            validation_split=0.25, 
            verbose=0,  # 关闭默认输出
            callbacks=[progress_callback]
        )
        
        # 评估模型
        test_loss, test_acc = model.evaluate(X_test_reshaped, y_test, verbose=0)
        
        # 保存模型（使用新的.keras格式以避免警告）
        model.save(f'./Model/{model_name}.keras')
        
        # 记录结果
        training_time = time.time() - start_time
        result = {
            'model_name': model_name,
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'total_params': total_params,
            'training_time': training_time
        }
        training_results.append(result)
        
        print("-" * 80)
        print(f"✅ 模型 {model_name} 训练完成!")
        print(f"   🎯 测试准确率: {test_acc:.4f}")
        print(f"   📉 测试损失: {test_loss:.4f}")
        print(f"   ⏱️  训练时间: {training_time:.1f}秒")
        print(f"   💾 已保存到: ./Model/{model_name}.keras")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ 模型 {model_name} 训练失败: {str(e)}")
        result = {
            'model_name': model_name,
            'test_accuracy': 0,
            'test_loss': float('inf'),
            'total_params': 0,
            'training_time': 0,
            'error': str(e)
        }
        training_results.append(result)
        
    finally:
        overall_pbar.update(1)

overall_pbar.close()

print("\n" + "🎉" * 30)
print("🏆 所有模型训练完成! 🏆")
print("🎉" * 30)

# 显示训练结果摘要
print("\n📊 训练结果排行榜:")
print("=" * 120)
print(f"{'排名':<6} {'模型名称':<25} {'🎯准确率':<12} {'📉损失':<12} {'📊参数量':<15} {'⏱️训练时间(s)':<15}")
print("=" * 120)

# 按准确率排序
successful_results = [r for r in training_results if 'error' not in r]
successful_results.sort(key=lambda x: x['test_accuracy'], reverse=True)

for idx, result in enumerate(successful_results[:40], 1):  # 显示所有成功的模型
    rank_emoji = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else f"{idx:2d}"
    print(f"{rank_emoji:<6} {result['model_name']:<25} {result['test_accuracy']:<12.4f} "
          f"{result['test_loss']:<12.4f} {result['total_params']:<15,} "
          f"{result['training_time']:<15.1f}")

# 显示失败的模型
failed_results = [r for r in training_results if 'error' in r]
if failed_results:
    print(f"\n❌ 失败的模型 ({len(failed_results)}个):")
    for result in failed_results:
        print(f"   {result['model_name']}: {result['error']}")

print(f"\n📁 已保存的模型文件 ({len(successful_results)}个):")
for result in successful_results:
    print(f"   ./Model/{result['model_name']}.keras")

if successful_results:
    print(f"\n🏆 最佳模型: {successful_results[0]['model_name']} (准确率: {successful_results[0]['test_accuracy']:.4f})")
    print("🎊" * 50)
