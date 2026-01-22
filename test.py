import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import cross_val_score



#laoding DF
df = pd.read_csv('ESS.csv')




#remove surplus columns
df_clean = df.drop(['name','essround','edition','proddate','idno','dweight','anweight','badge','prob', 'stratum','psu','pspwght','pweight'], axis=1)
#rename columns 
df_clean=df_clean.rename(columns={'bctprd':'boycott', 'gndr':'gender', 'agea':'age','cntry':'country'})
df_clean['boycott']=df_clean['boycott'].replace({2:0})
df_clean = df_clean[df_clean['boycott'].isin([0,1])]
#map missing values & clean up missing value

human_value_cols = ['ipcrtiva','impfreea','impdiffa','iplylfra','ipgdtima','impfuna',
                    'ipshabta','ipadvnta','impricha','ipsucesa','iprspota','impsafea','impenva',
                    'ipfrulea','ipudrsta','ipmodsta','imptrada','ipstrgva','ipbhprpa','ipeqopta','iphlppla']

hv_missing_vals = [66, 77, 88, 99]

for col in human_value_cols:
    df_clean[col]=df_clean[col].replace(hv_missing_vals, np.nan)

df_clean ['gender']   = df_clean ['gender'].replace([9], np.nan)
df_clean ['age']   = df_clean ['age'].replace([999], np.nan)
df_clean ['eduyrs'] = df_clean ['eduyrs'].replace([77,88,99], np.nan)

df_clean = df_clean.dropna(axis=0, thresh=0.2*df.shape[1])


#MI
cols_to_impute = df_clean.columns.drop('country','boycott')
imputer = IterativeImputer(max_iter=10, random_state=42)
df_clean[cols_to_impute] = imputer.fit_transform(df_clean[cols_to_impute])




#aggregating
df_clean['selfDirection'] = df_clean[['ipcrtiva','impfreea']].mean(axis=1)
df_clean['stimulation'] = df_clean[['impdiffa','iplylfra']].mean(axis=1)
df_clean['hedonism'] = df_clean[['ipgdtima','impfuna']].mean(axis=1)
df_clean['achievement'] = df_clean[['ipshabta','ipadvnta']].mean(axis=1)
df_clean['power'] = df_clean[['impricha','ipsucesa','iprspota']].mean(axis=1)
df_clean['security'] = df_clean[['impsafea','impenva']].mean(axis=1)
df_clean['conformity'] = df_clean['ipfrulea']  # only one item
df_clean['tradition'] = df_clean[['ipudrsta','ipmodsta','imptrada']].mean(axis=1)
df_clean['benevolence'] = df_clean[['ipstrgva','ipbhprpa']].mean(axis=1)
df_clean['universalism'] = df_clean[['ipeqopta','iphlppla']].mean(axis=1)

#one-hot

#define target, featuers
target = df_clean['boycott'].astype(int)
features= pd.get_dummies(df_clean.drop(['boycott'], axis=1), columns=['country'], drop_first=True)


#train test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target)

#model training


from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

pipe_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=2000, solver='lbfgs', random_state=42))
])
pipe_lr.fit(X_train, y_train)

# Predicted probabilities
y_prob_lr = pipe_lr.predict_proba(X_test)[:, 1]

# Predicted labels
y_pred_lr = pipe_lr.predict(X_test)


# Compute precision, recall
precision, recall, _ = precision_recall_curve(y_test, y_prob_lr)

# Compute PR-AUC
pr_auc_lr = average_precision_score(y_test, y_prob_lr)

# Plot PR curve with highlighted area
plt.figure(figsize=(6, 4))
plt.plot(recall, precision, color='green', lw=2, label='PR Curve')
plt.fill_between(recall, precision, alpha=0.2, color='lightgreen')  # highlight area
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Logistic Regression Precision–Recall Curve (PR-AUC = {pr_auc_lr*100:.1f}%)")
plt.grid(True)
plt.legend()
plt.show()

# Print metrics
y_pred_lr = pipe_lr.predict(X_test)
print('accuracy_lr:', accuracy_score(y_test, y_pred_lr))
print('f1_lr:', f1_score(y_test, y_pred_lr))
print('pr_auc_lr:', pr_auc_lr)


#RF
rf = RandomForestClassifier()
rf = rf.fit(X_train,y_train)
y_pred_rf = rf.predict(X_test)
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# Use predicted probabilities for the positive class
y_prob_rf = rf.predict_proba(X_test)[:, 1]

# Compute precision, recall, thresholds
precision, recall, _ = precision_recall_curve(y_test, y_prob_rf)

# Compute PR-AUC
pr_auc_rf = average_precision_score(y_test, y_prob_rf)

# Plot PR curve with highlighted area
plt.figure(figsize=(6, 4))
plt.plot(recall, precision, color='blue', lw=2, label='PR Curve')
plt.fill_between(recall, precision, alpha=0.2, color='skyblue')  # highlight area
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Random Forest Precision–Recall Curve (PR-AUC = {pr_auc_rf*100:.1f}%)")
plt.grid(True)
plt.legend()
plt.show()

# Print metrics
y_pred_rf = rf.predict(X_test)
print('accuracy_rf:', accuracy_score(y_test, y_pred_rf))
print('f1_rf:', f1_score(y_test, y_pred_rf))
print('pr_auc_rf:', pr_auc_rf)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

# --------------------
# Majority baseline predictions
# --------------------
# 全部預測為 0（負類）
y_pred_majority = np.zeros_like(y_test)

# --------------------
# Metrics
# --------------------
accuracy_majority = accuracy_score(y_test, y_pred_majority)
f1_majority = f1_score(y_test, y_pred_majority, zero_division=0)

# PR-AUC for majority / random baseline
pos_ratio = np.mean(y_test)
pr_auc_majority = pos_ratio

# --------------------
# Majority baseline PR curve
# --------------------
recall = np.linspace(0, 1, 100)
precision = np.full_like(recall, pos_ratio)

# --------------------
# Plot
# --------------------
plt.figure(figsize=(6, 4))

plt.plot(
    recall, precision,
    color='salmon', lw=2,
    label=f'Majority baseline (PR-AUC = {pr_auc_majority*100:.1f}%)'
)
plt.fill_between(recall, precision, alpha=0.2, color='salmon')

plt.xlim(0, 1)
plt.ylim(0, 1.05)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Majority (Random) Baseline Precision–Recall Curve")
plt.grid(True)
plt.legend()
plt.show()

# --------------------
# Print stats
# --------------------
print('accuracy_majority:', accuracy_majority)
print('f1_majority:', f1_majority)
print('pr_auc_majority:', pr_auc_majority)

