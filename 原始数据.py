import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import (confusion_matrix, roc_curve, auc, precision_recall_curve,
                             precision_score, recall_score, f1_score, accuracy_score)
from sklearn.preprocessing import label_binarize
import seaborn as sns
import os

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

    # 假设第一列为标签，第二列及之后为特征
    X = train_data.iloc[:, 3:]
    y = train_data.iloc[:, 0]

    # 确保标签只有1-7类
    if len(np.unique(y)) > 7:
        raise ValueError("训练数据中包含超过7类的标签，请检查数据！")

    # 加载外部验证数据集
    external_data = pd.read_excel(external_path)
    print("\n外部验证集缺失值统计：")
    print(external_data.isnull().sum())

    # 假设外部验证集与训练集结构相同
    X_external = external_data.iloc[:, 3:]
    y_external = external_data.iloc[:, 0]

    # 确保外部验证集的标签只有1-7类
    if len(np.unique(y_external)) > 7:
        raise ValueError("外部验证数据中包含超过7类的标签，请检查数据！")

    return X, y, X_external, y_external


# ---------------------------
# 2. 定义Stacking模型
# ---------------------------
def create_stacking_model(random_state=42):
    """创建Stacking集成模型"""
    # 定义基模型
    base_models = [
        ('ada', AdaBoostClassifier(random_state=random_state)),
        ('knn', KNeighborsClassifier()),
        ('lr', LogisticRegression(multi_class='multinomial',
                                  random_state=random_state,
                                  max_iter=1000)),
        ('rf', RandomForestClassifier(random_state=random_state)),
        ('svm', SVC(probability=True, random_state=random_state)),
        ('xgb', XGBClassifier(objective='multi:softprob',
                              random_state=random_state,
                              use_label_encoder=False,
                              eval_metric='mlogloss'))
    ]

    # 定义元模型
    meta_model = LogisticRegression(multi_class='multinomial',
                                    random_state=random_state,
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
# 3. 评估指标计算与保存
# ---------------------------
def calculate_metrics(y_true, y_pred, y_proba, classes, dataset_name, save_dir):
    """计算评估指标并保存到Excel"""
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_true, y_pred)

    # 计算各项指标
    metrics = {
        'Class': classes,
        'Accuracy': [accuracy_score(y_true, y_pred)] * len(classes),
        'Sensitivity': recall_score(y_true, y_pred, average=None),
        'Specificity': [],
        'PPV (Precision)': precision_score(y_true, y_pred, average=None),
        'NPV': [],
        'F1-score': f1_score(y_true, y_pred, average=None)
    }

    # 计算特异性和阴性预测值
    for i in range(len(classes)):
        tn = np.sum(np.delete(np.delete(conf_matrix, i, axis=0), i, axis=1))
        fp = np.sum(conf_matrix[:, i]) - conf_matrix[i, i]
        metrics['Specificity'].append(tn / (tn + fp))

        fn = np.sum(conf_matrix[i, :]) - conf_matrix[i, i]
        metrics['NPV'].append(tn / (tn + fn))

    # 计算ROC AUC和PR AUC
    y_one_hot = label_binarize(y_true, classes=classes)
    roc_auc = []
    pr_auc = []
    for i in range(len(classes)):
        # ROC AUC
        fpr, tpr, _ = roc_curve(y_one_hot[:, i], y_proba[:, i])
        roc_auc.append(auc(fpr, tpr))

        # PR AUC
        precision_pr, recall_pr, _ = precision_recall_curve(y_one_hot[:, i], y_proba[:, i])
        pr_auc.append(auc(recall_pr, precision_pr))

    metrics['ROC AUC'] = roc_auc
    metrics['PR AUC'] = pr_auc

    # 创建DataFrame
    metrics_df = pd.DataFrame(metrics)

    # 计算宏平均和加权平均
    macro_avg = metrics_df.iloc[:, 2:].mean()
    weighted_avg = metrics_df.iloc[:, 2:].apply(
        lambda x: np.average(x, weights=np.bincount(y_true)[classes]))

    # 添加平均行
    metrics_df.loc[len(metrics_df)] = ['Macro Avg'] + [np.nan] + macro_avg.tolist()
    metrics_df.loc[len(metrics_df)] = ['Weighted Avg'] + [np.nan] + weighted_avg.tolist()

    # 保存到Excel
    save_path = os.path.join(save_dir, f"{dataset_name}_metrics.xlsx")
    metrics_df.to_excel(save_path, index=False)
    print(f"\n{dataset_name}评估指标已保存至: {save_path}")

    return metrics_df, conf_matrix


def save_combined_metrics(test_metrics, external_metrics, save_dir):
    """将测试集和验证集指标合并保存到一张Excel表中"""
    # 复制数据框避免修改原数据
    test_df = test_metrics.copy()
    external_df = external_metrics.copy()

    # 添加数据集标识列
    test_df.insert(0, 'Dataset', 'Test Set')
    external_df.insert(0, 'Dataset', 'External Validation Set')

    # 合并数据框
    combined_df = pd.concat([test_df, external_df], axis=0)

    # 保存到Excel
    save_path = os.path.join(save_dir, 'combined_metrics.xlsx')
    combined_df.to_excel(save_path, index=False)
    print(f"整合评估指标已保存至: {save_path}")

    return combined_df


# ---------------------------
# 4. 可视化函数
# ---------------------------
def plot_confusion_matrix(conf_matrix, classes, title, save_path):
    """绘制混淆矩阵热力图"""
    plt.figure(figsize=(10, 5), dpi=300)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGnBu', cbar_kws={'shrink': 0.75})
    plt.xlabel('Predicted Label', fontsize=7)
    plt.ylabel('True Label', fontsize=7)
    plt.title(title, fontsize=8)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_roc_curve(y_true, y_proba, classes, title, save_path):
    """绘制ROC曲线"""
    plt.figure(figsize=(10, 5), dpi=300)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', '#FFA500']
    y_one_hot = label_binarize(y_true, classes=classes)

    # 计算每类ROC曲线
    fpr_dict = {}
    tpr_dict = {}
    roc_auc_dict = {}
    for i in range(len(classes)):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_one_hot[:, i], y_proba[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
        plt.plot(fpr_dict[i], tpr_dict[i], color=colors[i % len(colors)], lw=0.8,
                 label=f'Class {classes[i]} (AUC = {roc_auc_dict[i]:.3f})')

    # 计算并绘制Macro-average ROC曲线
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(len(classes))]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= len(classes)
    macro_auc = auc(all_fpr, mean_tpr)
    plt.plot(all_fpr, mean_tpr, color='k', linestyle='--', lw=1.5,
             label=f'Macro-average (AUC = {macro_auc:.3f})')

    # 绘制随机线
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='Random (AUC = 0.500)')

    plt.xlabel('False Positive Rate', fontsize=7)
    plt.ylabel('True Positive Rate', fontsize=7)
    plt.title(title, fontsize=8)
    plt.legend(loc='lower right', fontsize=5)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_pr_curve(y_true, y_proba, classes, title, save_path):
    """绘制PR曲线"""
    plt.figure(figsize=(10, 5), dpi=300)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', '#FFA500']
    y_one_hot = label_binarize(y_true, classes=classes)

    for i in range(len(classes)):
        precision, recall, _ = precision_recall_curve(y_one_hot[:, i], y_proba[:, i])
        plt.plot(recall, precision, color=colors[i % len(colors)], lw=0.8,
                 label=f'Class {classes[i]} PR Curve')

    plt.xlabel('Recall', fontsize=7)
    plt.ylabel('Precision', fontsize=7)
    plt.title(title, fontsize=8)
    plt.legend(loc='lower left', fontsize=5)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


# ---------------------------
# 5. 主函数
# ---------------------------
def main():
    # 数据路径
    train_path = 'E:/10EYE/7分类/结果/最佳-adaboost+knn+lr+rf+svm+xgboost/特征排序/发现集-填补-排序-14233-14F.xlsx'
    external_path = 'E:/10EYE/7分类/结果/最佳-adaboost+knn+lr+rf+svm+xgboost/特征排序/内部验证集-填补-排序-14F.xlsx'
    save_dir = 'D:/RS/10EYE/10分类/results1'

    # 1. 加载数据
    X, y, X_external, y_external = load_data(train_path, external_path)
    classes = np.unique(y)

    # 2. 划分训练集和测试集 (70%训练，30%测试)
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
        y_test, y_pred_test, y_proba_test, classes, "test", save_dir)

    # 可视化测试集结果
    plot_confusion_matrix(test_conf_matrix, classes,
                          'Confusion Matrix (Test Set)',
                          os.path.join(save_dir, 'Test_Confusion_Matrix.png'))

    plot_roc_curve(y_test, y_proba_test, classes,
                   'Stacking Classification ROC Curves (Test Set)',
                   os.path.join(save_dir, 'Test_ROC_Curve.png'))

    plot_pr_curve(y_test, y_proba_test, classes,
                  'Stacking Classification PR Curves (Test Set)',
                  os.path.join(save_dir, 'Test_PR_Curve.png'))

    # 5. 外部验证
    print("\n在完整训练集上重新训练模型并进行外部验证...")
    stacking_clf.fit(X, y)  # 使用全部训练数据重新训练

    # 预测外部验证集
    y_pred_external = stacking_clf.predict(X_external)
    y_proba_external = stacking_clf.predict_proba(X_external)

    # 计算外部验证集指标并保存
    external_metrics, external_conf_matrix = calculate_metrics(
        y_external, y_pred_external, y_proba_external, classes, "external", save_dir)

    # 保存整合的评估指标
    save_combined_metrics(test_metrics, external_metrics, save_dir)

    # 可视化外部验证结果
    plot_confusion_matrix(external_conf_matrix, classes,
                          'Confusion Matrix (External Validation Set)',
                          os.path.join(save_dir, 'External_Confusion_Matrix.png'))

    plot_roc_curve(y_external, y_proba_external, classes,
                   'Stacking Classification ROC Curves (External Validation Set)',
                   os.path.join(save_dir, 'External_ROC_Curve.png'))

    plot_pr_curve(y_external, y_proba_external, classes,
                  'Stacking Classification PR Curves (External Validation Set)',
                  os.path.join(save_dir, 'External_PR_Curve.png'))

    print("\n所有评估已完成，结果已保存至:", save_dir)


if __name__ == "__main__":
    main()