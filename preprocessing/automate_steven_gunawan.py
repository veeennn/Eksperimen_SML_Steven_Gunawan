from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv("kc_house_data_raw.csv")
# Menentukan fitur numerik dan kategoris
numeric_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_features = data.select_dtypes(include=['object']).columns.tolist()
column_names = data.columns
# Mendapatkan nama kolom tanpa kolom target
column_names = data.columns.drop("id")

# Membuat DataFrame kosong dengan nama kolom
df_header = pd.DataFrame(columns=column_names)

# Pastikan target_column tidak ada di numeric_features atau categorical_features
if target_column in numeric_features:
    numeric_features.remove(target_column)
if target_column in categorical_features:
    categorical_features.remove(target_column)

# Pipeline untuk fitur numerik
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Pipeline untuk fitur kategoris
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Column Transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)
dfApa = preprocessor.fit_transform(data)
dfApa.tocsv("ty.csv")
