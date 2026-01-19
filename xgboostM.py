import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, average_precision_score
from xgboost import XGBClassifier

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------
df = pd.read_csv('ESS.csv')

# --------------------------------------------------
# 2. Keep only boycott classes 1 and 2
# --------------------------------------------------
df_filtered = df[df['bctprd'].isin([1, 2])].copy()

# --------------------------------------------------
# 3. Keep only valid value responses (1–6)
# --------------------------------------------------
value_cols = [
    'ipcrtiva','impfreea','impdiffa','iplylfra','ipgdtima','impfuna',
    'ipshabta','ipadvnta','impricha','ipsucesa','iprspota',
    'impsafea','impenva','ipfrulea',
    'ipudrsta','ipmodsta','imptrada',
    'ipstrgva','ipbhprpa','ipeqopta','iphlppla'
]

df_filtered[value_cols] = df_filtered[value_cols].applymap(
    lambda x: x if x in [1,2,3,4,5,6] else pd.NA
)

# --------------------------------------------------
# 4. Consolidate Schwartz value constructs (10)
# --------------------------------------------------
df_filtered.loc[:, 'SelfDirection'] = df_filtered[['ipcrtiva','impfreea']].mean(axis=1)
df_filtered.loc[:, 'Stimulation'] = df_filtered[['impdiffa','iplylfra']].mean(axis=1)
df_filtered.loc[:, 'Hedonism'] = df_filtered[['ipgdtima','impfuna']].mean(axis=1)
df_filtered.loc[:, 'Achievement'] = df_filtered[['ipshabta','ipadvnta']].mean(axis=1)
df_filtered.loc[:, 'Power'] = df_filtered[['impricha','ipsucesa','iprspota']].mean(axis=1)
df_filtered.loc[:, 'Security'] = df_filtered[['impsafea','impenva']].mean(axis=1)
df_filtered.loc[:, 'Conformity'] = df_filtered['ipfrulea']
df_filtered.loc[:, 'Tradition'] = df_filtered[['ipudrsta','ipmodsta','imptrada']].mean(axis=1)
df_filtered.loc[:, 'Benevolence'] = df_filtered[['ipstrgva','ipbhprpa']].mean(axis=1)
df_filtered.loc[:, 'Universalism'] = df_filtered[['ipeqopta','iphlppla']].mean(axis=1)

# --------------------------------------------------
# 5. Select features and apply listwise deletion
# --------------------------------------------------
features = [
    'SelfDirection','Stimulation','Hedonism','Achievement','Power',
    'Security','Conformity','Tradition','Benevolence','Universalism',
    'agea','gndr'
]

df_model = df_filtered[features + ['bctprd']].dropna()

# --------------------------------------------------
# 6. Define X and y
# Map boycott: 1 -> 0 (minority), 2 -> 1 (majority)
# --------------------------------------------------
X = df_model[features]
y = df_model['bctprd'].map({1:0, 2:1})

# --------------------------------------------------
# 7. Stratified train-test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --------------------------------------------------
# 8. Feature scaling
# --------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# 9. Handle class imbalance
# --------------------------------------------------
scale_pos_weight = (y_train == 1).sum() / (y_train == 0).sum()

xgb = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

# --------------------------------------------------
# 10. Fit model and predict
# --------------------------------------------------
xgb.fit(X_train_scaled, y_train)

y_pred = xgb.predict(X_test_scaled)
y_pred_prob = xgb.predict_proba(X_test_scaled)[:, 1]

# --------------------------------------------------
# 11. Evaluation
# --------------------------------------------------
print("XGBoost F1 (minority class):",
      f1_score(y_test, y_pred, pos_label=0))

print("XGBoost Weighted F1:",
      f1_score(y_test, y_pred, average='weighted'))

print("Confusion Matrix:\n",
      confusion_matrix(y_test, y_pred))

print("PR-AUC:",
      average_precision_score(y_test, y_pred_prob))


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, average_precision_score
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# 1. Features and target (reuse from preprocessing)
# --------------------------------------------------
features = [
    'SelfDirection','Stimulation','Hedonism','Achievement','Power',
    'Security','Conformity','Tradition','Benevolence','Universalism',
    'agea','gndr'
]

X = df_model[features]
y = df_model['bctprd'].map({1:0, 2:1})  # boycott = 0

# --------------------------------------------------
# 2. Stratified split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --------------------------------------------------
# 3. Scale features (MANDATORY for SVM)
# --------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# 4. SVM with class weighting
# --------------------------------------------------
svm = SVC(
    kernel='rbf',
    class_weight='balanced',
    probability=True,
    random_state=42
)

# --------------------------------------------------
# 5. Fit and predict
# --------------------------------------------------
svm.fit(X_train_scaled, y_train)
y_pred = svm.predict(X_test_scaled)
y_pred_prob = svm.predict_proba(X_test_scaled)[:, 1]

# --------------------------------------------------
# 6. Evaluation
# --------------------------------------------------
print("SVM F1 (minority class):",
      f1_score(y_test, y_pred, pos_label=0))

print("SVM Weighted F1:",
      f1_score(y_test, y_pred, average='weighted'))

print("Confusion Matrix:\n",
      confusion_matrix(y_test, y_pred))

print("PR-AUC:",
      average_precision_score(y_test, y_pred_prob))
