import pandas as pd
import glob
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor


yildiz_geohash = 'sxk9s1'
seed = 43

path = './data/trafik' # use your path
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    print(filename)
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

df_fact = pd.concat(li, axis=0, ignore_index=True)

path = 'data/meteoroloji' # use your path
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    print(filename)
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

df_met = pd.concat(li, axis=0, ignore_index=True)

df_fact = df_fact[df_fact['GEOHASH'] == yildiz_geohash]
df_met = df_met[df_met['OBSERVATORY_NAME'] == 'BESIKTAS_YILDIZ']
df_merged = df_fact.merge(df_met, how='inner', on='DATE_TIME')
df_merged.sort_values(by='DATE_TIME', ascending=True)
df_merged['date_time_conv'] = df_merged['DATE_TIME'].apply(lambda x: dt.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
df_merged.drop(labels=['DATE_TIME', 'GEOHASH', 'OBSERVATORY_NAME', 'SENSOR_TYPE', '_id'], axis=1, inplace=True)

df_merged['year'], df_merged['month'], df_merged['day'], df_merged['hour'] = df_merged['date_time_conv'].dt.year, df_merged['date_time_conv'].dt.month, \
    df_merged['date_time_conv'].dt.day, df_merged['date_time_conv'].dt.hour

df_merged.drop(labels=['date_time_conv'], axis=1, inplace=True)
train_df = df_merged[df_merged['year'] == 2020]
test_df = df_merged[df_merged['year'] == 2021]

scaler = MinMaxScaler()

train_df[['MAXIMUM_WIND', 'AVERAGE_WIND', 'hour', 'month', 'day']] = scaler.fit_transform(train_df[['MAXIMUM_WIND', 'AVERAGE_WIND', 'hour', 'month', 'day']])

train_df = train_df[['AVERAGE_SPEED', 'MAXIMUM_WIND', 'AVERAGE_WIND', 'hour', 'month', 'day']]

test_df = test_df[['AVERAGE_SPEED', 'MAXIMUM_WIND', 'AVERAGE_WIND', 'hour', 'month', 'day']]

y_train = train_df[['AVERAGE_SPEED']].values.ravel()
X_train = train_df.drop(labels=['AVERAGE_SPEED'], axis=1)
y_test = test_df[['AVERAGE_SPEED']].values.ravel()
X_test = test_df.drop(labels=['AVERAGE_SPEED'], axis=1)

gb = GradientBoostingRegressor(random_state = seed)

gb.fit(X_train, y_train)

# Saving the model
import pickle
pickle.dump(gb, open('arac_tahmin.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))



