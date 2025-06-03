import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from keras.layers import Layer, Lambda
from sklearn.metrics import (confusion_matrix, classification_report, 
                           roc_curve, auc, precision_recall_curve,
                           accuracy_score, precision_score, recall_score, 
                           f1_score, cohen_kappa_score, matthews_corrcoef)
from sklearn.preprocessing import LabelEncoder, label_binarize
from keras.utils import to_categorical
import keras.backend as K
import warnings
import os
from tqdm import tqdm
import time
from keras.layers import Dense

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False


class MambaBlock(Layer):
    def __init__(self, d_model=128, **kwargs):
        super(MambaBlock, self).__init__(**kwargs)
        self.d_model = d_model
        
    def build(self, input_shape):
        from keras.layers import Dense
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
    
    def get_config(self):
        config = super(MambaBlock, self).get_config()
        config.update({"d_model": self.d_model})
        return config
    
class LinearAttentionLayer(Layer):
    """修复后的线性注意力层 - 数值稳定版本"""
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
        
        # 数值稳定的线性化 - 避免exp溢出
        # 方法1: 使用softplus替代exp
        q = K.softplus(q) + 1e-8  # 添加小常数避免零
        k = K.softplus(k) + 1e-8
        
        # 或者方法2: 使用ReLU + 1
        # q = K.relu(q) + 1.0
        # k = K.relu(k) + 1.0
        
        # 计算注意力
        kv = K.batch_dot(k, v, axes=[1, 1])
        qkv = K.batch_dot(q, kv, axes=[2, 1])
        
        return qkv
    
    def get_config(self):
        config = super(LinearAttentionLayer, self).get_config()
        config.update({"units": self.units})
        return config

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
    
    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            "num_patches": self.num_patches,
            "d_model": self.d_model
        })
        return config

# 定义自定义对象字典
custom_objects = {
    'MambaBlock': MambaBlock,
    'LinearAttentionLayer': LinearAttentionLayer,
    'PositionalEncoding': PositionalEncoding,
}

# 创建保存结果的目录
os.makedirs('./Evaluation_Results', exist_ok=True)
os.makedirs('./Evaluation_Results/ROC_Curves', exist_ok=True)
os.makedirs('./Evaluation_Results/Confusion_Matrices', exist_ok=True)
os.makedirs('./Evaluation_Results/PR_Curves', exist_ok=True)
os.makedirs('./Evaluation_Results/Performance_Comparison', exist_ok=True)

# 加载数据
print("📊 加载测试数据...")
file_path = './Data/data.pkl'
with open(file_path, 'rb') as f:
    data = pickle.load(f)
    
X = np.array(list(zip(*data))[0])
y = np.array(list(zip(*data))[1])

# 应用与训练时相同的数据处理
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

sampling_strategy_over = {'Pneumonia': 150, 'Healthy': 150, 'URTI': 100, 
                         'Bronchiectasis': 100, 'Bronchiolitis': 100}
smote = SMOTE(sampling_strategy=sampling_strategy_over, k_neighbors=5, random_state=42)
X_resampled_over, y_resampled_over = smote.fit_resample(X, y)

copd_indices = np.where(y_resampled_over == 'COPD')[0]
np.random.seed(42)
np.random.shuffle(copd_indices)
remove_indices = copd_indices[:len(copd_indices) // 2]

X_resampled = np.delete(X_resampled_over, remove_indices, axis=0)
y_resampled = np.delete(y_resampled_over, remove_indices)

# 编码标签
le = LabelEncoder()
y_encoded = le.fit_transform(y_resampled)
y_categorical = to_categorical(y_encoded)
class_names = le.classes_

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_categorical, 
                                                    test_size=0.2, random_state=42)
X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 189, 1))

# 获取真实标签（非one-hot）
y_test_labels = np.argmax(y_test, axis=1)

# 模型名称列表
model_names = [
    'basic_cnn', 'deep_cnn', 'batch_norm_cnn', 'lstm', 'bilstm', 'gru', 
    'cnn_lstm', 'cnn_gru', 'residual_cnn', 'multiscale', 'dense_cnn', 
    'separable_cnn', 'attention_lstm', 'wide_deep_cnn', 'pyramid_cnn', 
    'stacked_lstm', 'dilated_cnn', 'locally_connected', 'elu_cnn', 
    'swish_cnn', 'l1_regularized', 'l2_regularized', 'double_conv', 
    'global_max_pool', 'avg_pool', 'mamba_inspired', 'transformer_like',
    'pure_mamba', 'mamba_transformer', 'vit_like', 'bert_like', 'gpt_like',
    'sparse_attention', 'cross_attention', 'self_attention',
    'hierarchical_transformer', 'conv_transformer', 'mega_lstm', 'hybrid_mamba'
]

def plot_confusion_matrix(y_true, y_pred, model_name, class_names):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'混淆矩阵 - {model_name}', fontsize=16, pad=20)
    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'./Evaluation_Results/Confusion_Matrices/{model_name}_confusion_matrix.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 计算归一化混淆矩阵
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'归一化混淆矩阵 - {model_name}', fontsize=16, pad=20)
    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'./Evaluation_Results/Confusion_Matrices/{model_name}_normalized_confusion_matrix.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curves(y_true, y_score, model_name, class_names):
    """绘制多分类ROC曲线"""
    n_classes = len(class_names)
    
    # 将标签二值化
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    
    # 计算每个类别的ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 计算微平均ROC曲线和AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # 计算宏平均ROC曲线和AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # 绘制所有ROC曲线
    plt.figure(figsize=(12, 10))
    
    # 绘制每个类别的ROC曲线
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
    for i, color in enumerate(colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
    
    # 绘制微平均和宏平均
    plt.plot(fpr["micro"], tpr["micro"],
            label=f'微平均 (AUC = {roc_auc["micro"]:.3f})',
            color='deeppink', linestyle=':', linewidth=3)
    
    plt.plot(fpr["macro"], tpr["macro"],
            label=f'宏平均 (AUC = {roc_auc["macro"]:.3f})',
            color='navy', linestyle=':', linewidth=3)
    
    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='随机分类器')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率 (FPR)', fontsize=12)
    plt.ylabel('真阳性率 (TPR)', fontsize=12)
    plt.title(f'ROC曲线 - {model_name}', fontsize=16, pad=20)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'./Evaluation_Results/ROC_Curves/{model_name}_roc_curves.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return roc_auc

def plot_pr_curves(y_true, y_score, model_name, class_names):
    """绘制多分类PR曲线"""
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    
    precision = dict()
    recall = dict()
    pr_auc = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        pr_auc[i] = auc(recall[i], precision[i])
    
    # 绘制PR曲线
    plt.figure(figsize=(12, 10))
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
    
    for i, color in enumerate(colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                label=f'{class_names[i]} (AUC = {pr_auc[i]:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('召回率 (Recall)', fontsize=12)
    plt.ylabel('精确率 (Precision)', fontsize=12)
    plt.title(f'PR曲线 - {model_name}', fontsize=16, pad=20)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'./Evaluation_Results/PR_Curves/{model_name}_pr_curves.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def calculate_metrics(y_true, y_pred, y_score):
    """计算各种性能指标"""
    # 基础指标
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro')
    precision_weighted = precision_score(y_true, y_pred, average='weighted')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    recall_weighted = recall_score(y_true, y_pred, average='weighted')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    # 高级指标
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # 每个类别的指标
    class_report = classification_report(y_true, y_pred, output_dict=True)
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'cohen_kappa': kappa,
        'matthews_corrcoef': mcc,
        'class_report': class_report
    }

def load_model_safely(model_path, model_name):
    """安全加载模型，处理Lambda层和自定义层"""
    try:
        # 首先尝试使用safe_mode=False加载（处理Lambda层）
        model = load_model(model_path, custom_objects=custom_objects, safe_mode=False)
        return model
    except Exception as e1:
        try:
            # 如果失败，尝试仅使用custom_objects加载
            model = load_model(model_path, custom_objects=custom_objects)
            return model
        except Exception as e2:
            # 如果还是失败，根据模型名称添加特定的Lambda函数
            if 'swish' in model_name:
                def swish(x):
                    return x * K.sigmoid(x)
                custom_objects['swish'] = swish
            
            try:
                model = load_model(model_path, custom_objects=custom_objects, safe_mode=False)
                return model
            except Exception as e3:
                raise Exception(f"无法加载模型 {model_name}: {str(e3)}")

# 存储所有模型的评估结果
all_results = []

print("\n🚀 开始评估所有模型...")
print("="*80)

# 评估每个模型
for model_name in tqdm(model_names, desc="评估进度"):
    model_path = f'./Model/{model_name}.keras'
    
    if not os.path.exists(model_path):
        print(f"\n⚠️  模型文件不存在: {model_path}")
        continue
    
    try:
        print(f"\n📊 正在评估模型: {model_name}")
        
        # 安全加载模型
        model = load_model_safely(model_path, model_name)
        
        # 预测
        start_time = time.time()
        y_pred_proba = model.predict(X_test_reshaped, verbose=0)
        inference_time = time.time() - start_time
        
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # 计算性能指标
        metrics = calculate_metrics(y_test_labels, y_pred, y_pred_proba)
        
        # 绘制混淆矩阵
        plot_confusion_matrix(y_test_labels, y_pred, model_name, class_names)
        
        # 绘制ROC曲线
        roc_auc_scores = plot_roc_curves(y_test_labels, y_pred_proba, model_name, class_names)
        
        # 绘制PR曲线
        plot_pr_curves(y_test_labels, y_pred_proba, model_name, class_names)
        
        # 整理结果
        result = {
            'model_name': model_name,
            'accuracy': metrics['accuracy'],
            'precision_macro': metrics['precision_macro'],
            'precision_weighted': metrics['precision_weighted'],
            'recall_macro': metrics['recall_macro'],
            'recall_weighted': metrics['recall_weighted'],
            'f1_macro': metrics['f1_macro'],
            'f1_weighted': metrics['f1_weighted'],
            'cohen_kappa': metrics['cohen_kappa'],
            'matthews_corrcoef': metrics['matthews_corrcoef'],
            'auc_macro': roc_auc_scores['macro'],
            'auc_micro': roc_auc_scores['micro'],
            'inference_time_total': inference_time,
            'inference_time_per_sample': inference_time / len(X_test_reshaped),
            'total_params': model.count_params()
        }
        
        # 添加每个类别的指标
        for class_name in class_names:
            if class_name in metrics['class_report']:
                result[f'{class_name}_precision'] = metrics['class_report'][class_name]['precision']
                result[f'{class_name}_recall'] = metrics['class_report'][class_name]['recall']
                result[f'{class_name}_f1-score'] = metrics['class_report'][class_name]['f1-score']
        
        all_results.append(result)
        
        print(f"✅ {model_name} 评估完成: Accuracy={metrics['accuracy']:.4f}, AUC={roc_auc_scores['macro']:.4f}")
        
    except Exception as e:
        print(f"❌ 评估模型 {model_name} 时出错: {str(e)}")
        # 记录失败的模型
        result = {
            'model_name': model_name,
            'error': str(e),
            'accuracy': 0,
            'auc_macro': 0
        }
        all_results.append(result)

# 保存评估结果到CSV
if all_results:
    # 筛选出成功评估的结果
    successful_results = [r for r in all_results if 'error' not in r]
    
    if successful_results:
        df_results = pd.DataFrame(successful_results)
        df_results = df_results.sort_values('accuracy', ascending=False)
        df_results.to_csv('./Evaluation_Results/model_evaluation_results.csv', index=False)
        print(f"\n💾 评估结果已保存到: ./Evaluation_Results/model_evaluation_results.csv")
        
        # 显示前10个模型的结果
        print("\n🏆 Top 10 模型性能排名:")
        print("="*100)
        print(df_results[['model_name', 'accuracy', 'f1_weighted', 'auc_macro', 
                         'cohen_kappa', 'inference_time_per_sample']].head(10).to_string(index=False))

# 绘制性能比较图
def plot_performance_comparison(df_results):
    """绘制模型性能比较图"""
    
    # 1. 准确率对比条形图
    plt.figure(figsize=(15, 10))
    df_sorted = df_results.sort_values('accuracy', ascending=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_sorted)))
    
    plt.barh(df_sorted['model_name'], df_sorted['accuracy'], color=colors)
    plt.xlabel('准确率', fontsize=12)
    plt.title('模型准确率对比', fontsize=16, pad=20)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('./Evaluation_Results/Performance_Comparison/accuracy_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 多指标雷达图（前5个模型）
    top_models = df_results.head(5)
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='polar')
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    for idx, row in top_models.iterrows():
        values = [
            row['accuracy'],
            row['precision_weighted'],
            row['recall_weighted'],
            row['f1_weighted'],
            row['auc_macro']
        ]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['model_name'])
        ax.fill(angles, values, alpha=0.15)
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('Top 5 模型多维度性能对比', fontsize=16, pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('./Evaluation_Results/Performance_Comparison/radar_chart_top5.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 性能vs推理时间散点图
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df_results['inference_time_per_sample'] * 1000, 
                         df_results['accuracy'],
                         c=df_results['total_params'],
                         s=100, alpha=0.6, cmap='coolwarm')
    
    plt.xlabel('推理时间 (ms/样本)', fontsize=12)
    plt.ylabel('准确率', fontsize=12)
    plt.title('模型准确率 vs 推理时间', fontsize=16, pad=20)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('模型参数量', fontsize=10)
    
    # 标注前5个模型
    for idx, row in df_results.head(5).iterrows():
        plt.annotate(row['model_name'], 
                    (row['inference_time_per_sample'] * 1000, row['accuracy']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('./Evaluation_Results/Performance_Comparison/accuracy_vs_inference_time.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 各类别F1分数热力图
    class_columns = [col for col in df_results.columns if col.endswith('_f1-score')]
    if class_columns:
        class_f1_data = df_results[['model_name'] + class_columns].set_index('model_name')
        class_f1_data.columns = [col.replace('_f1-score', '') for col in class_f1_data.columns]
        
        plt.figure(figsize=(12, 15))
        sns.heatmap(class_f1_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                   cbar_kws={'label': 'F1-Score'})
        plt.title('各模型在不同类别上的F1分数', fontsize=16, pad=20)
        plt.xlabel('疾病类别', fontsize=12)
        plt.ylabel('模型名称', fontsize=12)
        plt.tight_layout()
        plt.savefig('./Evaluation_Results/Performance_Comparison/class_f1_heatmap.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. 综合性能得分
    # 计算综合得分（可以根据需求调整权重）
    df_results['composite_score'] = (
        0.3 * df_results['accuracy'] + 
        0.2 * df_results['f1_weighted'] + 
        0.2 * df_results['auc_macro'] + 
        0.15 * df_results['precision_weighted'] + 
        0.15 * df_results['recall_weighted']
    )
    
    plt.figure(figsize=(15, 10))
    df_sorted = df_results.sort_values('composite_score', ascending=True)
    
    # 创建颜色映射
    colors = ['red' if score < 0.7 else 'orange' if score < 0.8 else 'green' 
              for score in df_sorted['composite_score']]
    
    plt.barh(df_sorted['model_name'], df_sorted['composite_score'], color=colors)
    plt.xlabel('综合性能得分', fontsize=12)
    plt.title('模型综合性能评分（加权）', fontsize=16, pad=20)
    plt.axvline(x=0.8, color='k', linestyle='--', alpha=0.3, label='优秀阈值')
    plt.axvline(x=0.7, color='k', linestyle=':', alpha=0.3, label='良好阈值')
    plt.legend()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('./Evaluation_Results/Performance_Comparison/composite_score_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

# 只处理成功评估的结果
successful_results = [r for r in all_results if 'error' not in r]
if successful_results:
    df_results = pd.DataFrame(successful_results)
    df_results = df_results.sort_values('accuracy', ascending=False)
    plot_performance_comparison(df_results)
    print("\n📊 性能比较图已生成完成!")

# 生成评估报告摘要
def generate_evaluation_summary(df_results, all_results):
    """生成评估报告摘要"""
    summary = []
    summary.append("="*80)
    summary.append("模型评估报告摘要")
    summary.append("="*80)
    
    # 统计信息
    total_models = len(model_names)
    successful_models = len(df_results)
    failed_models = len([r for r in all_results if 'error' in r])
    
    summary.append(f"\n📊 评估统计:")
    summary.append(f"   - 总模型数: {total_models}")
    summary.append(f"   - 成功评估: {successful_models}")
    summary.append(f"   - 失败评估: {failed_models}")
    
    if len(df_results) > 0:
        # 最佳模型
        best_accuracy = df_results.iloc[0]
        summary.append(f"\n🏆 准确率最高模型: {best_accuracy['model_name']}")
        summary.append(f"   - 准确率: {best_accuracy['accuracy']:.4f}")
        summary.append(f"   - F1分数: {best_accuracy['f1_weighted']:.4f}")
        summary.append(f"   - AUC值: {best_accuracy['auc_macro']:.4f}")
        
        # 最快模型
        fastest = df_results.loc[df_results['inference_time_per_sample'].idxmin()]
        summary.append(f"\n⚡ 推理速度最快模型: {fastest['model_name']}")
        summary.append(f"   - 推理时间: {fastest['inference_time_per_sample']*1000:.2f} ms/样本")
        summary.append(f"   - 准确率: {fastest['accuracy']:.4f}")
        
        # 最小模型
        smallest = df_results.loc[df_results['total_params'].idxmin()]
        summary.append(f"\n📦 参数量最少模型: {smallest['model_name']}")
        summary.append(f"   - 参数量: {smallest['total_params']:,}")
        summary.append(f"   - 准确率: {smallest['accuracy']:.4f}")
        
        # 平均性能
        summary.append(f"\n📊 整体性能统计:")
        summary.append(f"   - 平均准确率: {df_results['accuracy'].mean():.4f} (±{df_results['accuracy'].std():.4f})")
        summary.append(f"   - 平均F1分数: {df_results['f1_weighted'].mean():.4f} (±{df_results['f1_weighted'].std():.4f})")
        summary.append(f"   - 平均AUC值: {df_results['auc_macro'].mean():.4f} (±{df_results['auc_macro'].std():.4f})")
    
    # 失败的模型
    failed_results = [r for r in all_results if 'error' in r]
    if failed_results:
        summary.append(f"\n❌ 评估失败的模型 ({len(failed_results)}个):")
        for result in failed_results:
            summary.append(f"   - {result['model_name']}: {result['error'][:100]}...")
    
    # 保存摘要
    with open('./Evaluation_Results/evaluation_summary.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary))
    
    # 打印摘要
    print('\n'.join(summary))

if all_results:
    successful_results = [r for r in all_results if 'error' not in r]
    if successful_results:
        df_results = pd.DataFrame(successful_results)
        df_results = df_results.sort_values('accuracy', ascending=False)
        generate_evaluation_summary(df_results, all_results)

print("\n✅ 所有评估任务完成!")
print(f"📁 结果保存在: ./Evaluation_Results/")
print("   - model_evaluation_results.csv: 详细评估指标")
print("   - ROC_Curves/: ROC曲线图")
print("   - Confusion_Matrices/: 混淆矩阵图") 
print("   - PR_Curves/: PR曲线图")
print("   - Performance_Comparison/: 性能对比图")
print("   - evaluation_summary.txt: 评估报告摘要")
