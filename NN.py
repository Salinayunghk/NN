# %%
import pandas as pd
from keras.layers import Input, LSTM, Dense, Concatenate
from keras.models import Model
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

###plot
import numpy as np
import matplotlib.pyplot as plt

# %%
Train = pd.read_csv('WAL_MODEL_Train.csv')
Test = pd.read_csv('WAL_MODEL_Test.csv')
Valid = pd.read_csv('WAL_MODEL_Valid.csv')
COST_data = pd.read_csv('COST.csv')
KO_data = pd.read_csv('KO.csv')
PEP_data = pd.read_csv('PEP.csv')
PG_data = pd.read_csv('PG.csv')
NSRGY_data = pd.read_csv('NSRGY.csv')

# %%
df = pd.concat([Train,Test,Valid],ignore_index=True)
df['Date'] = pd.to_datetime(df['Date'], format='%Y/%m/%d')


# %%
for all_date in [COST_data, KO_data, PEP_data, PG_data, NSRGY_data]:
    all_date['Date'] = pd.to_datetime(all_date['Date'], format='%d/%m/%Y %H:%M:%S').dt.date


# %%
print(['COST', 'KO', 'PEP', 'PG', 'NSRGY']+df.columns[1:].to_list())

# %%
prices = pd.concat([COST_data['Close'], KO_data['Close'], PEP_data['Close'], PG_data['Close'], NSRGY_data['Close']],axis=1, ignore_index=True)
df = pd.concat([prices, df], axis=1, names=['COST', 'KO', 'PEP', 'PG', 'NSRGY']+df.columns[1:].to_list())

# %%
df = df.rename(columns={0:'COST', 1: "KO", 2: "PEP", 3:"PG", 4:'NSRGY'})

# %%
df = df.drop(labels=['Unnamed: 0'], axis=1)

# %%
### number of nan
df.isnull().sum()

# %%
### use pervious values
df = df.fillna(method='backfill')

# %%
df[df['Date'] < pd.to_datetime('2022-10-03')].isnull().sum()

# %%
df[df['Date'] < pd.to_datetime('2022-10-03')].to_csv('dataset.csv')

# %%
df_v1 = df[df['Date'] < pd.to_datetime('2022-10-03')]

# %%
df_v1['CPI'] = df_v1['CPI'].str[:3].astype(float)

# %%
list_df_v1 = df_v1.columns

# %%
WINDOW_SIZE = 3
for col in list_df_v1:
  for i in range(WINDOW_SIZE): # Shift values for each step in WINDOW_SIZE
      df_v1[f"{col} +{i+1}"] = df_v1[f"{col}"].shift(periods=i+1)
df_v1.head()
     

# %%
scaler = MinMaxScaler(feature_range = (0,1))
COST = scaler.fit_transform(df_v1['COST'].to_numpy().reshape(-1, 1))
scaler = MinMaxScaler(feature_range = (0,1))
KO = scaler.fit_transform(df_v1['KO'].to_numpy().reshape(-1, 1))

scaler = MinMaxScaler(feature_range = (0,1))
Yr10 = scaler.fit_transform(df_v1['10 Yr'].to_numpy().reshape(-1, 1))
scaler = MinMaxScaler(feature_range = (0,1))
DXY = scaler.fit_transform(df_v1['DXY_Close'].to_numpy().reshape(-1, 1))

scaler = MinMaxScaler(feature_range = (0,1))
CPI = scaler.fit_transform(df_v1['CPI'].to_numpy().reshape(-1, 1))
scaler = MinMaxScaler(feature_range = (0,1))
WAL = scaler.fit_transform(df_v1['WAL_Close'].to_numpy().reshape(-1, 1))


# %%
length_data = len(df_v1)     # rows that data has
split_ratio = 1           # %70 train + %30 validation
length_train = round(length_data * split_ratio)  

X_train_COST = []
y_train_COST = []

X_train_KO = []
y_train_KO = []

X_train_Yr10 = []
y_train_Yr10 = []

X_train_DXY = []
y_train_DXY = []

X_train_CPI = []
y_train_CPI = []

X_train_WAL = []
y_train_WAL = []


time_step = 3

for i in range(time_step, length_train):
    X_train_COST.append(COST[i-time_step:i])
    y_train_COST.append(COST[i])
    
    X_train_KO.append(KO[i-time_step:i])
    y_train_KO.append(KO[i])

    X_train_Yr10.append(Yr10[i-time_step:i])
    y_train_Yr10.append(Yr10[i])
    
    X_train_DXY.append(DXY[i-time_step:i])
    y_train_DXY.append(DXY[i])
    
    X_train_CPI.append(CPI[i-time_step:i])
    y_train_CPI.append(CPI[i])

    X_train_WAL.append(WAL[i-time_step:i])
    y_train_WAL.append(WAL[i])



# %%
np.array(X_train_COST).shape

# %%
# convert list to array
num = 4000
X1_train_COST, X1_train_KO, y1_train_COST, y1_train_KO = np.array(X_train_COST[:num]),np.array(X_train_KO[:num]), np.array(y_train_COST[:num]), np.array(y_train_KO[:num])
X1_train_Yr10, X1_train_DXY, y1_train_Yr10, y1_train_DXY = np.array(X_train_Yr10[:num]),np.array(X_train_DXY[:num]), np.array(y_train_Yr10[:num]), np.array(y_train_DXY[:num])
X1_train_CPI, X1_train_WAL, y1_train_CPI, y1_train_WAL = np.array(X_train_CPI[:num]),np.array(X_train_WAL[:num]), np.array(y_train_CPI[:num]), np.array(y_train_WAL[:num])


X_test_COST, y_test_COST, X_test_KO, y_test_KO = np.array(X_train_COST[num:]), np.array(y_train_COST[num:]), np.array(X_train_KO[num:]), np.array(y_train_KO[num:])
X_test_Yr10, y_test_Yr10, X_test_DXY, y_test_DXY = np.array(X_train_Yr10[num:]), np.array(y_train_Yr10[num:]), np.array(X_train_DXY[num:]), np.array(y_train_DXY[num:])
X_test_CPI, y_test_CPI, X_test_WAL, y_test_WAL = np.array(X_train_CPI[num:]), np.array(y_train_CPI[num:]), np.array(X_train_WAL[num:]), np.array(y_train_WAL[num:])


# %%
#X_train_COST[..., np.newaxis].shape

# %%
### PLOT graph
for tar in ['COST', 'KO', 'PEP', 'PG', 'NSRGY', 'WAL_Close', '6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', 'PCE', 'GDP',
       'DXY_Close', 'CRE', 'CPI', 'UNRATE']:
    plt.figure()
    plt.subplots(figsize = (15,6))
    plt.plot(df_v1.Date.values, df_v1[tar].values)
    plt.xlabel("Day1 to Day4051")
    plt.ylabel("Open Price")
    plt.title(tar+' values')
    plt.show()
    
    break

# %%
def lstm_sector(timesteps, units_l1, units_l2, X_train, y_train, X_val_dataset, y_val_data):

    # Define the inputs for each LSTM
    input1 = Input(shape=(timesteps, X_train[0].shape[-1]))
    lstm1 = LSTM(units_l1, return_sequences=True)(input1)
    lstm2 = LSTM(units_l2)(lstm1)

    input2 = Input(shape=(timesteps, X_train[1].shape[-1]))
    lstm3 = LSTM(units_l1, return_sequences=True)(input2)
    lstm4 = LSTM(units_l2)(lstm3)
    

    concat = Concatenate()([lstm2, lstm4])
    dense1 = Dense(16, activation='relu')(concat)
    output = Dense(1, activation='relu')(dense1)


    model = Model(inputs=[input1, input2], outputs=output)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics = [tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()])

    # Train the model
    trained_model = model.fit(x=X_train, y=y_train, validation_data = (X_val_dataset, y_val_data), batch_size=32, epochs=10)

    return trained_model



# %%
core_model = lstm_sector(timesteps = time_step, units_l1=32, units_l2=16,X_train=[X1_train_COST, 
X1_train_KO], y_train=y1_train_COST, X_val_dataset=[X_test_COST, 
X_test_KO], y_val_data = y_test_COST)

# %%
def lstm_sector_v1(timesteps, units_l1, units_l2, X_train, y_train, X_val_dataset, y_val_data):
    
    # Define the inputs for each LSTM
    input1 = Input(shape=(timesteps, X_train[0].shape[-1]))
    lstm1 = LSTM(units_l1, return_sequences=True)(input1)
    lstm2 = LSTM(units_l2)(lstm1)

    input2 = Input(shape=(timesteps, X_train[1].shape[-1]))
    lstm3 = LSTM(units_l1, return_sequences=True)(input2)
    lstm4 = LSTM(units_l2)(lstm3)
    
    input3 = Input(shape=(timesteps, X_train[2].shape[-1]))
    lstm5 = LSTM(units_l1, return_sequences=True)(input3)
    lstm6 = LSTM(units_l2)(lstm5)

    input4 = Input(shape=(timesteps, X_train[3].shape[-1]))
    lstm7 = LSTM(units_l1, return_sequences=True)(input4)
    lstm8 = LSTM(units_l2)(lstm7)

    input5 = Input(shape=(timesteps, X_train[4].shape[-1]))
    lstm9 = LSTM(units_l1, return_sequences=True)(input5)
    lstm10 = LSTM(units_l2)(lstm9)

    input6 = Input(shape=(timesteps, X_train[5].shape[-1]))
    lstm11 = LSTM(units_l1, return_sequences=True)(input6)
    lstm12 = LSTM(units_l2)(lstm11)

    concat = Concatenate()([lstm2, lstm4, lstm6, lstm8, lstm10, lstm12])
    dense1 = Dense(64, activation='relu')(concat)
    dense2 = Dense(32, activation='relu')(dense1)
    output = Dense(1, activation='relu')(dense2)


    model = Model(inputs=[input1, input2, input3, input4, input5, input6], outputs=output)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics = [tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()])

    # Train the model
    trained_model = model.fit(x=X_train, y=y_train, validation_data = (X_val_dataset, y_val_data), batch_size=32, epochs=10)

    return trained_model

# %%
def lstm_sector_v2(timesteps, units_l1, units_l2, X_train, y_train, X_val_dataset, y_val_data, optimizer, lr, batch_size, epoch):
    
    # Define the inputs for each LSTM
    input1 = Input(shape=(timesteps, X_train[0].shape[-1]))
    lstm1 = LSTM(units_l1, return_sequences=True)(input1)
    lstm2 = LSTM(units_l2)(lstm1)

    input2 = Input(shape=(timesteps, X_train[1].shape[-1]))
    lstm3 = LSTM(units_l1, return_sequences=True)(input2)
    lstm4 = LSTM(units_l2)(lstm3)
    
    input3 = Input(shape=(timesteps, X_train[2].shape[-1]))
    lstm5 = LSTM(units_l1, return_sequences=True)(input3)
    lstm6 = LSTM(units_l2)(lstm5)

    input4 = Input(shape=(timesteps, X_train[3].shape[-1]))
    lstm7 = LSTM(units_l1, return_sequences=True)(input4)
    lstm8 = LSTM(units_l2)(lstm7)

    input5 = Input(shape=(timesteps, X_train[4].shape[-1]))
    lstm9 = LSTM(units_l1, return_sequences=True)(input5)
    lstm10 = LSTM(units_l2)(lstm9)

    input6 = Input(shape=(timesteps, X_train[5].shape[-1]))
    lstm11 = LSTM(units_l1, return_sequences=True)(input6)
    lstm12 = LSTM(units_l2)(lstm11)

    concat = Concatenate()([lstm2, lstm4, lstm6, lstm8, lstm10, lstm12])
    dense1 = Dense(64, activation='relu')(concat)
    dense2 = Dense(32, activation='relu')(dense1)
    output = Dense(1, activation='relu')(dense2)


    model = Model(inputs=[input1, input2, input3, input4, input5, input6], outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics = [tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()])

    # Train the model
    trained_model = model.fit(x=X_train, y=y_train, validation_data = (X_val_dataset, y_val_data), batch_size=batch_size, epochs=epoch)

    return trained_model

# %%
core_model_v1 = lstm_sector_v1(timesteps = time_step, units_l1=32, units_l2=16
, X_train = [X1_train_COST, X1_train_KO, X1_train_Yr10, X1_train_DXY, X1_train_CPI, X1_train_WAL]
, y_train = y1_train_COST
, X_val_dataset=[X_test_COST, X_test_KO, X_test_Yr10, X_test_DXY, X_test_CPI, X_test_WAL]
, y_val_data = y_test_COST)

# %%
learning_rate = [0.1, 0.01, 0.001, 0.0005]
batch_size = [8, 16, 32, 64]
epochs = [10, 20, 30]
model_pool = []

for lr in learning_rate:
  optimizers = [tf.keras.optimizers.Adam(lr), tf.keras.optimizers.SGD(lr), tf.keras.optimizers.Nadam(lr), tf.keras.optimizers.RMSprop(lr)]
  for bs in batch_size:
    for epoch in epochs:
      for op in optimizers:
        model_pool.append(lstm_sector_v2(timesteps = time_step, units_l1=32, units_l2=16
                          , X_train = [X1_train_COST, X1_train_KO, X1_train_Yr10, X1_train_DXY, X1_train_CPI, X1_train_WAL]
                          , y_train = y1_train_COST
                          , X_val_dataset=[X_test_COST, X_test_KO, X_test_Yr10, X_test_DXY, X_test_CPI, X_test_WAL]
                          , y_val_data = y_test_COST
                          , optimizer = op
                          , lr = lr
                          , batch_size = bs
                          , epoch = epoch))

# %%



