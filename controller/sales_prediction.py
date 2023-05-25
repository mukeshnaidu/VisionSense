# import pandas as pd
# import numpy as np
# from prophet import Prophet
#
# sales = pd.read_csv('../assets/files/footfall_data.csv')
# sales
# sales.describe()
# print(min(sales['ds']))
# print(max(sales['ds']))
# store_sales = sales.groupby(['ds'],as_index=False).sum()
# store_sales = store_sales.rename(columns={'ds': 'ds','sales' : 'y'})
# store_sales
# train = store_sales[store_sales['ds']<'2023-05-23']
# test = store_sales[store_sales['ds']<='2023-05-23']
# m = Prophet()
# m.fit(train)
# future = m.make_future_dataframe(periods=60, freq='d')
# forecast = m.predict(future)
# forecast
# fig1 = m.plot(forecast)
# forecast_sub = forecast[['ds', 'yhat']]
# forecast_sub['ds'] = forecast_sub['ds'].astype(str)
# test_sub = test[['ds','y']]
# eval_df = test_sub.merge(forecast_sub,on=['ds'], how='left')
# eval_df['abs_error'] = abs(eval_df['y']-eval_df['yhat'])
# eval_df['daily_FA'] = 1-(eval_df['abs_error']/eval_df['y'])
#
# total_y = sum(eval_df['y'])
# total_error = sum(eval_df['abs_error'])
# forecast_acc = 1-(total_error/total_y)
# print(forecast_acc)