import pandas as pd
df = pd.read_csv('ESS.csv')
print(df.head())

print(df.columns)
print(df['cntry'].value_counts())

print(df.info())

target = df['bctprd']
features = df['age', 'gndr']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2) 

