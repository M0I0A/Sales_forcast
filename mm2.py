import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from keras.layers import LSTM
from sklearn.model_selection import train_test_split

store_sales = pd.read_csv('store_sale.csv')
store_sales.head(10)
	
# print(store_sales.info())
store_sales = store_sales.drop(['store','item'], axis=1)
store_sales['date'] = pd.to_datetime(store_sales['date'])
#date -> month
store_sales['date'] = store_sales['date'].dt.to_period('M')
monthly_sales = store_sales.groupby('date').sum().reset_index()

monthly_sales['date'] = monthly_sales['date'].dt.to_timestamp()
monthly_sales.head(10)
# plt.figure(figsize=(15,5))
# plt.plot(monthly_sales['date'], monthly_sales['sales'])
# plt.xlabel('Date')
# plt.xlabel('Sales')
# plt.title("Monthly Customer Sales")
# plt.show()
monthly_sales['sales_diff'] = monthly_sales['sales'].diff()
monthly_sales = monthly_sales.dropna()
monthly_sales.head(10)

#irstly, we need to drop off the ‘date’ and ‘sale’ columns in the dataset as we will be only dealing with the stationary sale data to train our machine learning model as well as the reinforcement learning model
	
supverised_data = monthly_sales.drop(['date','sales'], axis=1)

for i in range(1,13):
    col_name = 'month_' + str(i)
    supverised_data[col_name] = supverised_data['sales_diff'].shift(i)
supverised_data = supverised_data.dropna().reset_index(drop=True)
supverised_data.head(10)

#Now split this data into training and test data:
train_data = supverised_data[:-12]
test_data = supverised_data[-12:]
print('Train Data Shape:', train_data.shape)
print('Test Data Shape:', test_data.shape)

#The next step is to scale the feature values to restrict them to a range of (-1,1) using MinMaxScaler():
scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

#n the supervised DataFrame, first column corresponds to the output and the remaining columns act as the input features
X_train, y_train = train_data[:,1:], train_data[:,0:1]
X_test, y_test = test_data[:,1:], test_data[:,0:1]
y_train = y_train.ravel()
y_test = y_test.ravel()
print('X_train Shape:', X_train.shape)
print('y_train Shape:', y_train.shape)
print('X_test Shape:', X_test.shape)
print('y_test Shape:', y_test.shape)

#In the last step of data pre-processing, we will make a prediction data frame to merge the predicted sale prices of all the trained algorithms:
sales_dates = monthly_sales['date'][-12:].reset_index(drop=True)
predict_df = pd.DataFrame(sales_dates)

act_sales = monthly_sales['sales'][-13:].to_list()
#1.Forecast Sales using Linear Regression
linreg_model = LinearRegression()
linreg_model.fit(X_train, y_train)
linreg_pred = linreg_model.predict(X_test)
linreg_pred = linreg_pred.reshape(-1,1)
linreg_pred_test_set = np.concatenate([linreg_pred,X_test], axis=1)
linreg_pred_test_set = scaler.inverse_transform(linreg_pred_test_set)
result_list = []
for index in range(0, len(linreg_pred_test_set)):
    result_list.append(linreg_pred_test_set[index][0] + act_sales[index])
linreg_pred_series = pd.Series(result_list,name='linreg_pred')
pre = predict_df.merge(linreg_pred_series, left_index=True, right_index=True)
linreg_rmse = np.sqrt(mean_squared_error(pre['linreg_pred'], monthly_sales['sales'][-12:]))
linreg_mae = mean_absolute_error(pre['linreg_pred'], monthly_sales['sales'][-12:])
linreg_r2 = r2_score(pre['linreg_pred'], monthly_sales['sales'][-12:])
print('Linear Regression RMSE: ', linreg_rmse)
print('Linear Regression MAE: ', linreg_mae)
print('Linear Regression R2 Score: ', linreg_r2)
# plt.figure(figsize=(15,7))
# plt.plot(monthly_sales['date'], monthly_sales['sales'])
# plt.plot(pre['date'], pre['linreg_pred'])
# plt.title("Customer Sales Forecast using Linear Regression")
# plt.xlabel("Date")
# plt.ylabel("Sales")
# plt.legend(["Original Sales", "Predicted Sales"])
# plt.show()

#2.orecast Sales using Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, max_depth=20)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_pred = rf_pred.reshape(-1,1)
rf_pred_test_set = np.concatenate([rf_pred,X_test], axis=1)
rf_pred_test_set = scaler.inverse_transform(rf_pred_test_set)
result_list = []
for index in range(0, len(rf_pred_test_set)):
    result_list.append(rf_pred_test_set[index][0] + act_sales[index])
rf_pred_series = pd.Series(result_list, name='rf_pred')
rrr = predict_df.merge(rf_pred_series, left_index=True, right_index=True)
rf_rmse = np.sqrt(mean_squared_error(rrr['rf_pred'], monthly_sales['sales'][-12:]))
rf_mae = mean_absolute_error(rrr['rf_pred'], monthly_sales['sales'][-12:])
rf_r2 = r2_score(rrr['rf_pred'], monthly_sales['sales'][-12:])
print('Random Forest RMSE: ', rf_rmse)
print('Random Forest MAE: ', rf_mae)
print('Random Forest R2 Score: ', rf_r2)
# plt.figure(figsize=(15,7))
# plt.plot(monthly_sales['date'], monthly_sales['sales'])
# plt.plot(rrr['date'], rrr['rf_pred'])
# plt.title("Customer Sales Forecast using Random Forest")
# plt.xlabel("Date")
# plt.ylabel("Sales")
# plt.legend(["Original Sales", "Predicted Sales"])
# plt.show()

#3.orecast Sales using XGBoost Regressor
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.2, objective='reg:squarederror')
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_pred = xgb_pred.reshape(-1,1)
xgb_pred_test_set = np.concatenate([xgb_pred,X_test], axis=1)
xgb_pred_test_set = scaler.inverse_transform(xgb_pred_test_set)
result_list = []
for index in range(0, len(xgb_pred_test_set)):
    result_list.append(xgb_pred_test_set[index][0] + act_sales[index])
xgb_pred_series = pd.Series(result_list, name='xgb_pred')
rre = predict_df.merge(xgb_pred_series, left_index=True, right_index=True)
xgb_rmse = np.sqrt(mean_squared_error(rre['xgb_pred'], monthly_sales['sales'][-12:]))
xgb_mae = mean_absolute_error(rre['xgb_pred'], monthly_sales['sales'][-12:])
xgb_r2 = r2_score(rre['xgb_pred'], monthly_sales['sales'][-12:])
print('XG Boost RMSE: ', xgb_rmse)
print('XG Boost MAE: ', xgb_mae)
print('XG Boost R2 Score: ', xgb_r2)
# plt.figure(figsize=(15,7))
# plt.plot(monthly_sales['date'], monthly_sales['sales'])
# plt.plot(rre['date'], rre['xgb_pred'])
# plt.title("Customer Sales Forecast using XG Boost")
# plt.xlabel("Date")
# plt.ylabel("Sales")
# plt.legend(["Original Sales", "Predicted Sales"])
# plt.show()

#4.Forecast Sales using LSTM RNN
# model = Sequential()
# model.add(LSTM(4, batch_input_shape=(1, X_train.shape[1], X_test.shape[2])))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# checkpoint_filepath = os.getcwd()
# model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False, monitor='val_loss', mode='min', save_best_only=True)
# callbacks = [EarlyStopping(patience=5), model_checkpoint_callback]
# history = model.fit(X_train, y_train, epochs=200, batch_size=1, validation_data=(X_test, y_test), callbacks=callbacks)
# metrics_df = pd.DataFrame(history.history)
# print(metrics_df)
# plt.figure(figsize=(10,5))
# plt.plot(metrics_df.index, metrics_df.loss)
# plt.plot(metrics_df.index, metrics_df.val_loss)
# plt.title('Sales Forecast Model Loss over Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Mean Squared Error')
# plt.legend(['Training Loss', 'Validation Loss'])
# plt.show()

lstm_rmse,lstm_mae,lstm_r2 = 15494.296868853291,12533.464960899204,0.9911825545326522

#Coparing Forecast Sales using Machine Learning Algorithms
# linreg_stats = [linreg_rmse, linreg_mae, linreg_r2]
# rf_stats = [rf_rmse, rf_mae, rf_r2]
# xgb_stats = [xgb_rmse, xgb_mae, xgb_r2]
# # lstm_stats = [lstm_rmse, lstm_mae, lstm_r2]
# plt.figure(figsize=(15,7))
# plt.plot(linreg_stats)
# plt.plot(rf_stats)
# plt.plot(xgb_stats)
# # plt.plot(lstm_stats)
# plt.title("Model Comparison between Linear Regression, Random Forest, XG Boost")
# plt.xticks([0,1,2], labels=['RMSE','MAE','R2 Score'])
# plt.legend(["Linear Regression", "Random Forest", "XG Boost"])
# plt.show()

# rre['xgb_pred'], monthly_sales['sales'][-12:]

plt.figure(figsize=(15,7))
# plt.plot(monthly_sales['date'], monthly_sales['sales'][-12:])
plt.plot(pre['date'], pre['linreg_pred'],linewidth=1, color="red")
plt.plot(rrr['date'], rrr['rf_pred'],linewidth=1, color="green")
plt.plot(rre['date'], rre['xgb_pred'],linewidth=1, color="blue")
plt.title("Customer Sales Forecast using All methods")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(["Linear Regression", "Random Forest Regressor","XGBoost Regressor"])
plt.show()