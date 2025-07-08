import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import (confusion_matrix, roc_curve, auc, precision_recall_curve,
                             precision_score, recall_score, f1_score, accuracy_score,
                             roc_auc_score, average_precision_score)
from sklearn.preprocessing import label_binarize
from sklearn.utils import resample
from scipy.stats import bootstrap
import seaborn as sns
import os
import shap  # 新增SHAP库

# 设置字体和图形参数
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False


# ---------------------------
# 1. 数据加载与预处理
# ---------------------------
def load_data(train_path, external_path):
    """加载训练集和外部验证集"""
    # 加载训练数据集
    train_data = pd.read_excel(train_path)
    print("训练集缺失值统计：")
    print(train_data.isnull().sum())

    X = train_data.iloc[:, 4:]
    y = train_data.iloc[:, 0]

    # 确保标签只有0和1（二分类）
    if len(np.unique(y)) > 2:
        raise ValueError("训练数据中包含超过2类的标签，请检查数据！")

    # 加载外部验证数据集
    external_data = pd.read_excel(external_path)
    print("\n外部验证集缺失值统计：")
    print(external_data.isnull().sum())

    X_external = external_data.iloc[:, 4:]
    y_external = external_data.iloc[:, 0]

    # 确保外部验证集的标签只有0和1
    if len(np.unique(y_external)) > 2:
        raise ValueError("外部验证数据中包含超过2类的标签，请检查数据！")

    return X, y, X_external, y_external


# ---------------------------
# 2. 定义Stacking模型
# ---------------------------
def create_stacking_model(random_state=42):
    """创建Stacking集成模型"""
    # 定义基模型 m
    base_models = [
        ('knn', KNeighborsClassifier()),
        ('lr', LogisticRegression(max_iter=1000, random_state=42),),
        ('mlp', MLPClassifier(random_state=42),),
        ('rf', RandomForestClassifier(random_state=random_state)),
        ('xgb', XGBClassifier(objective='binary:logistic',  # 修改为二分类目标
                              random_state=random_state,
                              use_label_encoder=False,
                              eval_metric='logloss'))
    ]

    # 定义元模型
    meta_model = LogisticRegression(random_state=random_state,  # 移除多分类参数
                                    max_iter=1000)

    # 创建Stacking分类器
    stacking_clf = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        stack_method='auto',
        n_jobs=-1,
        passthrough=False,
        verbose=1
    )

    return stacking_clf


# ---------------------------
# 3. 评估指标计算与保存（二分类简化版）
# ---------------------------
def compute_metrics(y_true, y_pred, y_proba):
    """计算二分类指标"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'sensitivity': recall_score(y_true, y_pred),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'ppv': precision_score(y_true, y_pred),
        'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba[:, 1]),  # 使用正类的概率
        'pr_auc': average_precision_score(y_true, y_proba[:, 1])
    }
    return metrics


def bootstrap_metrics(y_true, y_pred, y_proba, n_bootstrap=1000):
    """自助法计算置信区间（二分类）"""
    n_samples = len(y_true)
    boot_results = {k: [] for k in ['accuracy', 'sensitivity', 'specificity',
                                    'ppv', 'npv', 'f1', 'roc_auc', 'pr_auc']}

    for _ in range(n_bootstrap):
        indices = resample(np.arange(n_samples))
        y_true_boot = y_true.iloc[indices] if isinstance(y_true, pd.Series) else y_true[indices]
        y_pred_boot = y_pred[indices]
        y_proba_boot = y_proba[indices]

        metrics = compute_metrics(y_true_boot, y_pred_boot, y_proba_boot)
        for k in metrics:
            boot_results[k].append(metrics[k])

    # 计算置信区间
    def calculate_ci(data, confidence_level=0.95):
        return bootstrap((np.array(data),), np.mean, confidence_level=confidence_level).confidence_interval

    ci_results = {}
    for metric in boot_results:
        ci = calculate_ci(boot_results[metric])
        ci_results[f'{metric}_lower'] = ci.low
        ci_results[f'{metric}_upper'] = ci.high

    return ci_results


def calculate_metrics(y_true, y_pred, y_proba, dataset_name, save_dir, n_bootstrap=1000):
    """计算评估指标并保存到Excel（二分类）"""
    os.makedirs(save_dir, exist_ok=True)

    # 计算指标
    metrics = compute_metrics(y_true, y_pred, y_proba)
    ci_results = bootstrap_metrics(y_true, y_pred, y_proba, n_bootstrap)

    # 构建结果DataFrame
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Sensitivity (Recall)', 'Specificity',
                   'PPV (Precision)', 'NPV', 'F1-score',
                   'ROC AUC', 'PR AUC'],
        'Value': [metrics['accuracy'], metrics['sensitivity'], metrics['specificity'],
                  metrics['ppv'], metrics['npv'], metrics['f1'],
                  metrics['roc_auc'], metrics['pr_auc']],
        'Lower_CI': [ci_results['accuracy_lower'], ci_results['sensitivity_lower'],
                     ci_results['specificity_lower'], ci_results['ppv_lower'],
                     ci_results['npv_lower'], ci_results['f1_lower'],
                     ci_results['roc_auc_lower'], ci_results['pr_auc_lower']],
        'Upper_CI': [ci_results['accuracy_upper'], ci_results['sensitivity_upper'],
                     ci_results['specificity_upper'], ci_results['ppv_upper'],
                     ci_results['npv_upper'], ci_results['f1_upper'],
                     ci_results['roc_auc_upper'], ci_results['pr_auc_upper']]
    })

    save_path = os.path.join(save_dir, f"{dataset_name}_metrics.xlsx")
    metrics_df.to_excel(save_path, index=False)
    print(f"\n{dataset_name}评估指标已保存至: {save_path}")

    return metrics_df, confusion_matrix(y_true, y_pred)


# ---------------------------
# 4. 可视化函数（二分类简化版）
# ---------------------------
def plot_confusion_matrix(conf_matrix, title, save_path):
    """绘制混淆矩阵热力图"""
    plt.figure(figsize=(6, 5), dpi=300)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.xlabel('Predicted Label', fontsize=10)
    plt.ylabel('True Label', fontsize=10)
    plt.title(title, fontsize=12)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_roc_curve(y_true, y_proba, title, save_path):
    """绘制ROC曲线（二分类）"""
    plt.figure(figsize=(8, 6), dpi=300)
    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])  # 使用正类的概率
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, color='b', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=10)
    plt.ylabel('True Positive Rate', fontsize=10)
    plt.title(title, fontsize=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_pr_curve(y_true, y_proba, title, save_path):
    """绘制PR曲线（二分类）"""
    plt.figure(figsize=(8, 6), dpi=300)
    precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
    pr_auc = auc(recall, precision)

    plt.plot(recall, precision, color='b', lw=2,
             label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall', fontsize=10)
    plt.ylabel('Precision', fontsize=10)
    plt.title(title, fontsize=12)
    plt.legend(loc="upper right", fontsize=10)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_shap_plots(model, X, y, feature_names, save_dir):
    """绘制SHAP散点图和瀑布图"""
    # 创建SHAP解释器
    explainer = shap.Explainer(model.named_estimators_['xgb'])  # 使用XGBoost模型进行解释
    shap_values = explainer(X)

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 创建SHAP结果子目录
    shap_dir = os.path.join(save_dir, 'SHAP_Results')
    os.makedirs(shap_dir, exist_ok=True)

    # 1. 为每个指定特征绘制SHAP散点图并保存为PDF
    for feature in feature_names:
        if feature in X.columns:
            plt.figure(figsize=(8, 6), dpi=300)
            shap.plots.scatter(shap_values[:, feature], show=False)
            plt.title(f'SHAP Scatter Plot for {feature}', fontsize=12)
            save_path = os.path.join(shap_dir, f'SHAP_Scatter_{feature}.pdf')
            plt.savefig(save_path, bbox_inches='tight', dpi=300, format='pdf')
            plt.close()
            print(f"SHAP散点图已保存至: {save_path}")
        else:
            print(f"警告: 特征 '{feature}' 不在数据集中")

    # 2. 绘制瀑布图 - 阳性样本10个
    positive_indices = np.where(y == 1)[0]
    if len(positive_indices) >= 10:
        selected_positive = np.random.choice(positive_indices, 10, replace=False)
        for i, idx in enumerate(selected_positive):
            plt.figure(figsize=(10, 8), dpi=300)
            shap.plots.waterfall(shap_values[idx], show=False)
            plt.title(f'SHAP Waterfall Plot (Positive Sample {i + 1})', fontsize=12)
            save_path = os.path.join(shap_dir, f'SHAP_Waterfall_Positive_{i + 1}.pdf')
            plt.savefig(save_path, bbox_inches='tight', dpi=300, format='pdf')
            plt.close()
            print(f"阳性样本瀑布图已保存至: {save_path}")
    else:
        print(f"警告: 只有 {len(positive_indices)} 个阳性样本，无法选择10个")

    # 3. 绘制瀑布图 - 阴性样本10个
    negative_indices = np.where(y == 0)[0]
    if len(negative_indices) >= 10:
        selected_negative = np.random.choice(negative_indices, 10, replace=False)
        for i, idx in enumerate(selected_negative):
            plt.figure(figsize=(10, 8), dpi=300)
            shap.plots.waterfall(shap_values[idx], show=False)
            plt.title(f'SHAP Waterfall Plot (Negative Sample {i + 1})', fontsize=12)
            save_path = os.path.join(shap_dir, f'SHAP_Waterfall_Negative_{i + 1}.pdf')
            plt.savefig(save_path, bbox_inches='tight', dpi=300, format='pdf')
            plt.close()
            print(f"阴性样本瀑布图已保存至: {save_path}")
    else:
        print(f"警告: 只有 {len(negative_indices)} 个阴性样本，无法选择10个")


# ---------------------------
# 5. 主函数
# ---------------------------
def main():
    # 数据路径
    train_path = 'E:/10EYE/2分类/3.knn+lr+mlp+rf+xgboost-最终/特征缩减/发现集-填补-排序-特征排序-12.xlsx'
    external_path = 'E:/10EYE/2分类/3.knn+lr+mlp+rf+xgboost-最终/特征缩减/外部验证集3-填补-排序-特征排序-12-1.xlsx'
    save_dir = 'E:/10EYE/2分类/results'

    # 指定要绘制SHAP散点图的特征
    target_features = ['TBIL', 'TBA', 'DBIL', 'GLU', 'LDH', 'TG',
                       'BUN', 'ALP', 'DD', 'ALB', 'RDW_SD', 'TC']

    # 1. 加载数据
    X, y, X_external, y_external = load_data(train_path, external_path)

    # 2. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    # 3. 创建并训练模型
    print("\n训练模型中...")
    stacking_clf = create_stacking_model(random_state=42)
    stacking_clf.fit(X_train, y_train)

    # 4. 在测试集上评估
    print("\n在测试集上评估模型...")
    y_pred_test = stacking_clf.predict(X_test)
    y_proba_test = stacking_clf.predict_proba(X_test)

    # 计算测试集指标并保存
    test_metrics, test_conf_matrix = calculate_metrics(
        y_test, y_pred_test, y_proba_test, "test", save_dir)

    # 可视化测试集结果
    plot_confusion_matrix(test_conf_matrix,
                          'Confusion Matrix (Test Set)',
                          os.path.join(save_dir, 'Test_Confusion_Matrix.png'))

    plot_roc_curve(y_test, y_proba_test,
                   'ROC Curve (Test Set)',
                   os.path.join(save_dir, 'Test_ROC_Curve.png'))

    plot_pr_curve(y_test, y_proba_test,
                  'PR Curve (Test Set)',
                  os.path.join(save_dir, 'Test_PR_Curve.png'))

    # 5. 外部验证
    print("\n在完整训练集上重新训练模型并进行外部验证...")
    stacking_clf.fit(X, y)  # 使用全部训练数据重新训练

    # 预测外部验证集
    y_pred_external = stacking_clf.predict(X_external)
    y_proba_external = stacking_clf.predict_proba(X_external)

    # 计算外部验证集指标并保存
    external_metrics, external_conf_matrix = calculate_metrics(
        y_external, y_pred_external, y_proba_external, "external", save_dir)

    # 可视化外部验证结果
    plot_confusion_matrix(external_conf_matrix,
                          'Confusion Matrix (External Validation Set)',
                          os.path.join(save_dir, 'External_Confusion_Matrix.png'))

    plot_roc_curve(y_external, y_proba_external,
                   'ROC Curve (External Validation Set)',
                   os.path.join(save_dir, 'External_ROC_Curve.png'))

    plot_pr_curve(y_external, y_proba_external,
                  'PR Curve (External Validation Set)',
                  os.path.join(save_dir, 'External_PR_Curve.png'))

    # 6. 绘制SHAP图
    print("\n绘制SHAP图...")
    plot_shap_plots(stacking_clf, X, y, target_features, save_dir)

    print("\n所有评估已完成，结果已保存至:", save_dir)


if __name__ == "__main__":
    main()
