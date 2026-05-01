import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def build_features_pipeline(numeric_cols, categorical_cols):
    """
    Cria o transformador de colunas reproduzível.
    """

    # Pipeline para números: trata nulos com mediana e escala
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy = 'median')),
        ('scaler',StandardScaler())
    ])

    # Pipeline para categorias: trata nulos e faz One-Hot Encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Junta tudo no preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    return preprocessor

def initial_cleaning(df):
    """
    Limpeza bruta que não entra no pipeline do sklearn (ex: tipos de dados).
    """

    df = df.copy()
    # Converte Total Charges para numérico
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
    # Remove nulos criticos antes de seguir
    df = df.dropna(subset=['Total Charges'])
    return df