import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.sparse import hstack
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau

data = pd.read_csv('merge(in).csv')

data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

deceased_df = data[data['icu_death'] == 1]
deceased_df_sorted = deceased_df.sort_values(by=['id', 'charttime'])
numeric_df = deceased_df_sorted.select_dtypes(include=[np.number])
numeric_df.drop(columns=['id'], inplace=True)

correlation_matrix = numeric_df.corr()
correlation_with_los_icu = correlation_matrix['los_icu'].sort_values(key=abs, ascending=False)

top_features = correlation_with_los_icu[1:31].index.tolist()
df_selected_features = deceased_df_sorted[top_features + ['los_icu']]

X = df_selected_features.drop('los_icu', axis=1)
y = df_selected_features['los_icu']

X.fillna(X.mean(), inplace=True)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=140, validation_split=0.2, batch_size=32, verbose=1)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r_squared = r2_score(y_test, predictions)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r_squared}')
