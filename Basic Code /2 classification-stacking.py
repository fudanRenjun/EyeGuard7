import pandas as pd
import numpy as np
import itertools

from sklearn.ensemble import StackingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# 1. 读取 Excel 数据
df = pd.read_excel("E:/10EYE/2分类/数据/发现集-填补-排序1.xlsx")

# 获取标签和特征
y = df.iloc[:, 0]  # 标签列
X = df.iloc[:, 4:]  # 特征列

# 2. 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# 3. 定义基学习器（保持原配置）
models = {
    "adaboost": AdaBoostClassifier(random_state=42),
    "knn": KNeighborsClassifier(),
    "lr": LogisticRegression(max_iter=1000, random_state=42),
    "mlp": MLPClassifier(random_state=42),
    "nb": GaussianNB(),
    "rf": RandomForestClassifier(random_state=42),
    "svm": SVC(probability=True, random_state=42),
    "xgboost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# 4. 遍历所有组合
results = []

for r in range(2, len(models) + 1):
    for comb in itertools.combinations(models.items(), r):
        estimators = list(comb)
        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000, random_state=42),
            cv=5,  # 五折交叉验证
            passthrough=False,
            n_jobs=-1
        )
        try:
            stacking.fit(X_train, y_train)
            # 二分类时使用正类的概率（第二列）
            y_pred_proba = stacking.predict_proba(X_test)[:, 1]
            roc = roc_auc_score(y_test, y_pred_proba)  # 移除非必需参数
        except Exception as e:
            print(f"组合 {'+'.join([name for name, _ in estimators])} 跳过，原因：{e}")
            continue

        model_names = "+".join([name for name, _ in estimators])
        results.append({"Models": model_names, "ROC_AUC": roc})
        print(f"组合 {model_names} 完成，ROC_AUC: {roc}")

# 5. 保存结果
results_df = pd.DataFrame(results)
results_df.to_excel("E:/10EYE/2分类/stacking_results.xlsx", index=False)

print("所有组合的 ROC_AUC 结果已保存")
