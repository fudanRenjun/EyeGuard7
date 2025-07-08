import pandas as pd
import numpy as np
import itertools
from time import time
import os

from sklearn.ensemble import StackingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 1. 读取 Excel 数据
df = pd.read_excel("D:/RS/10EYE/10分类/发现集-新-填充.xlsx")

# 第一列为标签列
y = df.iloc[:, 0]
# 从第四列开始为特征数据
X = df.iloc[:, 3:]

# 2. 划分训练集与测试集，分层抽样保证类别比例一致
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# 3. 定义每个模型的超参数网格
param_grids = {
    "adaboost": {
        'n_estimators': [50, 100, 200],  # number of estimators
        'learning_rate': [0.01, 0.1, 1.0]  # learning rate
    },
    "knn": {
        'n_neighbors': [3, 5, 7, 9],  # number of neighbors
        'weights': ['uniform', 'distance'],  # distance metric
        'p': [1, 2]  # 1 for manhattan, 2 for euclidean
    },
    "lr": {
        'C': [0.001, 0.01, 0.1, 1, 10],  # regularization strength
        'penalty': ['l2'],  # penalty type (l1 requires different solver)
        'solver': ['lbfgs', 'saga']  # optimization algorithm
    },
    "mlp": {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],  # hidden layer sizes
        'activation': ['relu', 'tanh'],  # activation function
        'alpha': [0.0001, 0.001, 0.01]  # L2 penalty (regularization term)
    },
    "nb": {
        'var_smoothing': [1e-9, 1e-8, 1e-7]  # smoothing parameter α
    },
    "rf": {
        'n_estimators': [100, 200, 300],  # number of trees
        'max_depth': [None, 10, 20],  # maximum depth
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    "svm": {
        'C': [0.1, 1, 10],  # regularization C
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']  # kernel coefficient γ
    },
    "xgboost": {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],  # max depth
        'learning_rate': [0.01, 0.1],  # learning rate
        'subsample': [0.8, 1.0],  # subsampling ratio
        'colsample_bytree': [0.8, 1.0]
    }
}

# 4. 定义模型创建函数（带超参数调优）
def get_tuned_model(model_name, X_train, y_train):
    """为给定模型执行网格搜索并返回最佳模型"""
    start_time = time()

    # 创建基础模型
    if model_name == "adaboost":
        base_model = AdaBoostClassifier(random_state=42)
    elif model_name == "knn":
        base_model = KNeighborsClassifier()
    elif model_name == "lr":
        base_model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_name == "mlp":
        base_model = MLPClassifier(random_state=42, max_iter=1000)
    elif model_name == "nb":
        base_model = GaussianNB()
    elif model_name == "rf":
        base_model = RandomForestClassifier(random_state=42)
    elif model_name == "svm":
        base_model = SVC(probability=True, random_state=42)
    elif model_name == "xgboost":
        base_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    else:
        raise ValueError(f"未知模型: {model_name}")

    # 创建网格搜索对象
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grids[model_name],
        scoring='accuracy',  # 使用准确率作为评估指标
        cv=3,  # 3折交叉验证
        n_jobs=-1,  # 使用所有CPU核心
        verbose=1
    )

    # 执行网格搜索
    print(f"\n开始为 {model_name} 执行网格搜索...")
    grid_search.fit(X_train, y_train)

    # 输出最佳参数
    print(f"{model_name} 最佳参数: {grid_search.best_params_}")
    print(f"{model_name} 最佳准确率: {grid_search.best_score_:.4f}")
    print(f"{model_name} 调优耗时: {time() - start_time:.2f}秒")

    return grid_search.best_estimator_


# 5. 调优所有模型
tuned_models = {}
for model_name in param_grids.keys():
    tuned_models[model_name] = get_tuned_model(model_name, X_train, y_train)

# 6. 遍历基学习器所有组合（组合大小至少为2）
results = []  # 存储每个组合的名称及对应性能指标
best_roc_auc = 0
best_combo = None

# 创建结果目录
results_dir = "D:/RS/10EYE/10分类/stacking_results"
os.makedirs(results_dir, exist_ok=True)


# 定义评估函数
def evaluate_model(model, X_test, y_test):
    """评估模型性能"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # 计算多分类 ROC_AUC (采用 ovR 策略，macro 平均)
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    return roc_auc, accuracy, f1


for r in range(2, len(tuned_models) + 1):
    print(f"\n评估 {r} 个模型的组合...")
    for comb in itertools.combinations(tuned_models.items(), r):
        # comb 为 ((name, estimator), ...)，将其转换为列表
        estimators = list(comb)
        model_names = [name for name, _ in estimators]
        combo_name = "+".join(model_names)

        # 构建 stacking 模型，meta learner 选用 LogisticRegression
        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000, random_state=42),
            cv=5,
            passthrough=False,
            n_jobs=-1
        )

        try:
            # 训练 stacking 模型
            print(f"训练组合: {combo_name}")
            stacking.fit(X_train, y_train)

            # 评估模型
            roc_auc, accuracy, f1 = evaluate_model(stacking, X_test, y_test)

            # 保存结果
            result = {
                "Models": combo_name,
                "ROC_AUC": roc_auc,
                "Accuracy": accuracy,
                "F1_Score": f1,
                "Num_Models": r
            }
            results.append(result)

            print(f"组合 {combo_name} 完成 | ROC_AUC: {roc_auc:.4f} | Accuracy: {accuracy:.4f} | F1: {f1:.4f}")

            # 检查是否是最佳组合
            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                best_combo = combo_name
                print(f"新最佳组合: {best_combo} | ROC_AUC: {best_roc_auc:.4f}")

        except Exception as e:
            print(f"组合 {combo_name} 跳过，原因：{e}")
            continue

# 7. 将所有组合结果保存到 Excel 文件
results_df = pd.DataFrame(results)

# 按ROC_AUC降序排序
results_df = results_df.sort_values(by='ROC_AUC', ascending=False)

# 保存结果
results_file = os.path.join(results_dir, "stacking_results_tuned.xlsx")
results_df.to_excel(results_file, index=False)

# 保存最佳组合信息
best_combo_file = os.path.join(results_dir, "best_combo.txt")
with open(best_combo_file, 'w') as f:
    f.write(f"最佳组合: {best_combo}\n")
    f.write(f"最佳ROC_AUC: {best_roc_auc:.4f}\n")

    # 获取最佳组合的详细结果
    best_result = results_df[results_df['Models'] == best_combo].iloc[0]
    f.write(f"准确率: {best_result['Accuracy']:.4f}\n")
    f.write(f"F1分数: {best_result['F1_Score']:.4f}\n")

print(f"\n所有组合的评估结果已保存到: {results_file}")
print(f"最佳组合信息已保存到: {best_combo_file}")
print(f"最佳组合: {best_combo} | ROC_AUC: {best_roc_auc:.4f}")
