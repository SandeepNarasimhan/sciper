from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error

data = pd.read_csv('/Users/sandeep/Desktop/Time series data/Car Dekho Dataset.csv')

X = data.drop(['selling_price'], axis=1)
y = data[['selling_price']]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1234)

train_data = pd.concat([X_train, y_train], axis=1)

## Pipeline
class data_type_convert(BaseEstimator, TransformerMixin):

    def __init__(self, col):
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X[self.col] = X[self.col].replace(' ', np.nan)
        X[self.col] = X[self.col].astype(float)

        return X

class to_dataframe(BaseEstimator, TransformerMixin):

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X = pd.DataFrame(X, columns=self.columns)
        return X.convert_dtypes()

class Multicolinearity(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.7):
        self.threshold = threshold
        self.features_to_keep = []

    def fit(self, X, y=None):
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("CorrelationReducer expects a pandas DataFrame.")

        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop = [column for column in upper.columns if any(upper[column] > self.threshold)]
        self.features_to_keep = [col for col in X.columns if col not in to_drop]
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.features_to_keep)  # Attempt recovery
        return X[self.features_to_keep].copy()

class OutlierRemover(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None, method='iqr', factor=1.5):
        self.columns = columns
        self.method = method
        self.factor = factor

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.select_dtypes(include='number').columns.tolist()
        return self

    def transform(self, X):
        X_clean = X.copy()
        for col in self.columns:
            if self.method == 'iqr':
                Q1 = X_clean[col].quantile(0.25)
                Q3 = X_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - self.factor * IQR
                upper = Q3 + self.factor * IQR
                X_clean = X_clean[(X_clean[col] >= lower) & (X_clean[col] <= upper)]
                X_clean = X_clean.reset_index(drop=True)
            else:
                raise ValueError("Currently only 'iqr' method is supported.")
        return X_clean.drop(['seats'], axis=1)


cols = ['mileage()', 'engine()', 'max_power(Bhp)', 'seats']
all_cols = cols + ['Age', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']

num_cols = ['mileage()', 'engine()', 'max_power(Bhp)', 'Age', 'km_driven']
cat_cols = ['fuel', 'seller_type', 'transmission', 'owner']

ct1 = ColumnTransformer([
    ('imputer', SimpleImputer(), cols),
], remainder='passthrough')

ct2 = ColumnTransformer([
    ('romove_multicolinearity', Multicollinearity(threshold=0.7), num_cols)
], remainder='passthrough')

ct3 = ColumnTransformer([
    ('numeric', StandardScaler(), num_cols),
    ('categorical', OneHotEncoder(), cat_cols)
])

class ColumnTransformerToDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):
        self.transformer.fit(X, y)
        # Get transformed feature names
        num_features = self.transformer.named_transformers_['numeric'].get_feature_names_out(num_cols)
        cat_features = self.transformer.named_transformers_['categorical'].get_feature_names_out(cat_cols)
        self.columns = list(num_features) + list(cat_features)
        return self

    def transform(self, X):
        transformed = self.transformer.transform(X)
        return pd.DataFrame(transformed, columns=self.columns, index=X.index)

pipe = Pipeline([
    ('type_convert', data_type_convert('max_power(Bhp)')),
    ('numeric_impute', ct1),
    ('to_dataframe', to_dataframe(all_cols)),
    #('remove_multicollinearity', ct2),
    #('OutlierRemover', OutlierRemover(factor=2)),
    ('preprocessing', ColumnTransformerToDataFrame(ct3)),
    ('Regression', LinearRegression())
])

model = TransformedTargetRegressor(
    regressor=pipe,
    func=np.log1p,
    inverse_func=np.expm1
)

model.fit(X_train, y_train)
pred = model.predict(X_test)
pred
pred = pred.ravel() if pred.ndim > 1 else pred

mean_squared_error(y_test, pred)
root_mean_squared_error(y_test, pred)

plt.figure()
sns.scatterplot(x=y_test['selling_price'], y=pred)


sns.scatterplot(x=pred, y=y_test['selling_price'] - pred)
