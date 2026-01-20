import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

#laoding DF
df = pd.read_csv('ESS.csv')

#EDA
# age
print(f"age%: \n{df['agea'].value_counts(normalize=True).get(999)}")

#gender
print(f"gender% : \n {df['gndr'].value_counts(normalize=True)}")

#eduyrs
print(f"education% : \n {df['eduyrs'].value_counts(normalize=True)[[77, 88, 99]]}")

print(df['eduyrs'].isin([77,88,99]).mean())


#Featuer enginerring
df_filtered = df[df['bctprd'].isin([1,2])].copy()
df_filtered['bctprd']=df_filtered['bctprd'].map({1:1,2:0})

target = df_filtered['bctprd']
# Assuming df is your dataframe
# Replace values outside 1-6 with NaN
cols = ['ipcrtiva','impfreea','impdiffa','iplylfra','ipgdtima','impfuna',
        'ipshabta','ipadvnta','impricha','ipsucesa','iprspota','impsafea','impenva',
        'ipfrulea','ipudrsta','ipmodsta','imptrada','ipstrgva','ipbhprpa','ipeqopta','iphlppla']

df_filtered[cols] = df[cols].applymap(lambda x: x if x in [1,2,3,4,5,6] else pd.NA)
# Create 10 consolidated variables
df_filtered['SelfDirection'] = df_filtered[['ipcrtiva','impfreea']].mean(axis=1)
df_filtered['Stimulation'] = df_filtered[['impdiffa','iplylfra']].mean(axis=1)
df_filtered['Hedonism'] = df_filtered[['ipgdtima','impfuna']].mean(axis=1)
df_filtered['Achievement'] = df_filtered[['ipshabta','ipadvnta']].mean(axis=1)
df_filtered['Power'] = df_filtered[['impricha','ipsucesa','iprspota']].mean(axis=1)
df_filtered['Security'] = df_filtered[['impsafea','impenva']].mean(axis=1)
df_filtered['Conformity'] = df_filtered['ipfrulea']  # only one item
df_filtered['Tradition'] = df_filtered[['ipudrsta','ipmodsta','imptrada']].mean(axis=1)
df_filtered['Benevolence'] = df_filtered[['ipstrgva','ipbhprpa']].mean(axis=1)
df_filtered['Universalism'] = df_filtered[['ipeqopta','iphlppla']].mean(axis=1)


features = df_filtered[['gndr', 'agea','SelfDirection','Stimulation','Hedonism','Achievement','Power','Security','Conformity','Tradition','Benevolence','Universalism']]


