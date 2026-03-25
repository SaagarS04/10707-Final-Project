import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train_gc = pd.read_csv('train_game_context_2020-05-12_2025-08-01.csv')
train_fp = pd.read_csv('train_first_pitch_2020-05-12_2025-08-01.csv')
test_gc = pd.read_csv('test_game_context_2025-08-01_2025-11-03.csv')
test_fp = pd.read_csv('test_first_pitch_2025-08-01_2025-11-03.csv')

X_train = pd.get_dummies(train_gc.drop(columns=['game_date','game_id']), drop_first=True).astype(float)
cols = X_train.columns
X_train = X_train.values
X_mean, X_std = X_train.mean(0), X_train.std(0)
X_std[X_std < 1e-8] = 1.0
X_train = (X_train - X_mean) / X_std

X_test = pd.get_dummies(test_gc.drop(columns=['game_date','game_id']), drop_first=True).astype(float)
X_test = X_test.reindex(columns=cols, fill_value=0).values
X_test = (X_test - X_mean) / X_std

y_train = train_fp['pitch_type'].fillna(train_fp['pitch_type'].mode()[0])
y_test = test_fp['pitch_type'].fillna(test_fp['pitch_type'].mode()[0])

print(f'NaN in X_train: {np.isnan(X_train).sum()}')
print(f'NaN in X_test: {np.isnan(X_test).sum()}')

lr = LogisticRegression(max_iter=1000, C=1.0)
lr.fit(X_train, y_train)
train_pred = lr.predict(X_train)
test_pred = lr.predict(X_test)
print(f'Logistic Regression train accuracy: {accuracy_score(y_train, train_pred):.2%}')
print(f'Logistic Regression test accuracy:  {accuracy_score(y_test, test_pred):.2%}')
print(f'Base rate (always FF):              {(y_test == "FF").mean():.2%}')
print(f'Test predictions distribution:')
print(pd.Series(test_pred).value_counts(normalize=True).head(5))
