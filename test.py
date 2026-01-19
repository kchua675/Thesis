import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

#laoding DF
df = pd.read_csv('ESS.csv')

#EDA
df.isnull().sum()


#Featuer enginerring
df_filtered = df[df['bctprd'].isin([1,2])]
target = df_filtered['bctprd']
# Assuming df is your dataframe
# Replace values outside 1-6 with NaN
cols = ['ipcrtiva','impfreea','impdiffa','iplylfra','ipgdtima','impfuna',
        'ipshabta','ipadvnta','impricha','ipsucesa','iprspota','impsafea','impenva',
        'ipfrulea','ipudrsta','ipmodsta','imptrada','ipstrgva','ipbhprpa','ipeqopta','iphlppla']

df_filtered[cols] = df[cols].applymap(lambda x: x if x in [1,2,3,4,5,6] else pd.NA)
# Create 10 consolidated variables
df_filtered['SelfDirection'] = df[['ipcrtiva','impfreea']].mean(axis=1)
df_filtered['Stimulation'] = df[['impdiffa','iplylfra']].mean(axis=1)
df_filtered['Hedonism'] = df[['ipgdtima','impfuna']].mean(axis=1)
df_filtered['Achievement'] = df[['ipshabta','ipadvnta']].mean(axis=1)
df_filtered['Power'] = df[['impricha','ipsucesa','iprspota']].mean(axis=1)
df_filtered['Security'] = df[['impsafea','impenva']].mean(axis=1)
df_filtered['Conformity'] = df['ipfrulea']  # only one item
df_filtered['Tradition'] = df[['ipudrsta','ipmodsta','imptrada']].mean(axis=1)
df_filtered['Benevolence'] = df[['ipstrgva','ipbhprpa']].mean(axis=1)
df_filtered['Universalism'] = df[['ipeqopta','iphlppla']].mean(axis=1)


features = df_filtered[['gndr', 'agea','SelfDirection','Stimulation','Hedonism','Achievement','Power','Security','Conformity','Tradition','Benevolence','Universalism']]

#train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target) 

#model build
#baseline
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(class_weight='balanced', random_state=42)
y_pred_lr=lr.fit(X_train, y_train).predict(X_test)
print("lr:", f1_score(y_test, y_pred_lr, pos_label=1))
print("lr:", confusion_matrix(y_test,y_pred_lr))

#rf
from sklearn.ensemble import RandomForestClassifier
rf =RandomForestClassifier(class_weight='balanced',random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

#evaluation
print("rf:", f1_score(y_test, y_pred, pos_label=1))
print("rf:", confusion_matrix(y_test,y_pred))


