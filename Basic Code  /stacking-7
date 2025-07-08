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
# 假设 Excel 文件名为 "data.xlsx"
df = pd.read_excel("D:/RS/10EYE/10分类/发现集-新-填充.xlsx")

# 第一列为标签列（假设无表头或表头已处理），
# 若存在表头可使用 df['label'] 或 df.iloc[:,0]
y = df.iloc[:, 0]

# 从第四列开始为特征数据（注意 Python 索引从0开始）
X = df.iloc[:, 3:]

# 2. 划分训练集与测试集，分层抽样保证类别比例一致
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# 3. 定义八个基学习器
models = {
    "adaboost": AdaBoostClassifier(random_state=42),
    "knn": KNeighborsClassifier(),
    "lr": LogisticRegression(max_iter=1000, random_state=42),
    "mlp": MLPClassifier(random_state=42),
    "nb": GaussianNB(),
    "rf": RandomForestClassifier(random_state=42),
    "svm": SVC(probability=True, random_state=42),
    "xgboost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

# 4. 遍历基学习器所有组合（组合大小至少为2）
results = []  # 存储每个组合的名称及对应 ROC_AUC

for r in range(2, len(models) + 1):
    for comb in itertools.combinations(models.items(), r):
        # comb 为 ((name, estimator), ...)，将其转换为列表
        estimators = list(comb)
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
            stacking.fit(X_train, y_train)
            # 获取测试集预测概率
            y_pred_proba = stacking.predict_proba(X_test)
            # 计算多分类 ROC_AUC (采用 ovR 策略，macro 平均)
            roc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
        except Exception as e:
            # 若当前组合因某些模型不支持 predict_proba 或其它原因出错，则跳过
            print(f"组合 {'+'.join([name for name, _ in estimators])} 跳过，原因：{e}")
            continue

        # 保存结果：组合名称及对应 ROC_AUC 值
        model_names = "+".join([name for name, _ in estimators])
        results.append({"Models": model_names, "ROC_AUC": roc})
        print(f"组合 {model_names} 完成，ROC_AUC: {roc}")

# 5. 将所有组合结果保存到 Excel 文件
results_df = pd.DataFrame(results)
results_df.to_excel("D:/RS/10EYE/10分类/stacking_results.xlsx", index=False)

print("所有组合的 ROC_AUC 结果已保存到 stacking_results.xlsx")
