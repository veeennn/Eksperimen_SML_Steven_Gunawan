from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("kc_house_data_raw.csv")
df = df.drop(columns=['id'])
data.to_csv("kc_house_data_preprocessing.csv")
