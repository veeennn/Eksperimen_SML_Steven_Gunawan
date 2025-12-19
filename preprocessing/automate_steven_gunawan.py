
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


#Type your code here
df = pd.read_csv("kc_house_data_raw.csv")


#string harus di encode
encoder = LabelEncoder()
kategori = df.select_dtypes(include=['object']).columns
numerik = df.select_dtypes(include=['float64', 'int64']).columns
for col in kategori:
  df[col] = encoder.fit_transform(df[col])
df[col].head()
df.drop_duplicates()



df.fillna(df.mean())
df.dropna()

scaler = MinMaxScaler()
df[numerik] = scaler.fit_transform(df[numerik])

for cols in numerik:
    Q1 = df[cols].quantile(0.25)
    Q3 = df[cols].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[cols] >= lower_bound) & (df[cols] <= upper_bound)]
df = df.drop(columns=['id'])

df.to_csv("kc_house_data_preprocessing.csv")
