import pandas as pd
import numpy as np
# from statsmodels.tsa.arima.model import ARIMA
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.croston import Croston
import matplotlib.pyplot as plt
from sktime.forecasting.arima import AutoARIMA
import time


class Instance:
    def __init__(self, sku):
        self.sku = sku
        self.df = pd.DataFrame()
        self.fc = None
        self.real = None
        self.index = None

    def init_df(self, df, overwrite=False):
        # initialise for SKU
        if len(self.df.index) == 0 or overwrite:
            # filter for SKU and convert to Time Series
            self.df = (df[df['Encoded_SKU_ID'] == SKU]).drop(columns=['Encoded_SKU_ID']).set_index('SALES_DATE')
            self.df = self.df.reindex(DATES, fill_value=0)
            self.df = self.df.squeeze()
            self.df = self.df.asfreq('D')

    def ARIMA_forecast(self, order=(1, 1, 1), seasonal_ord=(1, 1, 1, 12)):
        # give inputs of p,d,q and seasonal P,D,Q,S for ARIMA
        # depending on input can take long
        assert len(self.df.index) != 0, 'run self.init_df(df) first!'

        model = ARIMA(order=order, seasonal_order=seasonal_ord, suppress_warnings=True)
        model.fit(self.df.loc[:last_train_date])

        fc = model.predict(list(range(1, 8)))
        real = self.df.loc[last_train_date + np.timedelta64(1, 'D'):]
        self._save_(real, fc)

    def AutoARIMA_forecast(self):
        # hyer parameter tuning if ARIMA
        assert len(self.df.index) != 0, 'run self.init_df(df) first!'

        model = AutoARIMA(sp=12, max_p=3, max_q=3, max_d=3, suppress_warnings=True)
        model.fit(self.df.loc[:last_train_date])

        fc = model.predict(list(range(1, 8)))
        real = self.df.loc[last_train_date + np.timedelta64(1, 'D'):]
        self._save_(real, fc)

    def _save_(self, real, fc):
        # update data
        assert (real.index == fc.index).all(), 'check date index'

        self.index = fc.index
        self.fc = fc.to_numpy()
        self.real = real.to_numpy().reshape(fc.shape)

    def Croston_forecast(self, smooth=0.1):
        # Croston method
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8629246/
        # https://link.springer.com/article/10.1057/jors.1972.50
        assert len(self.df.index) != 0, 'run self.init_df(df) first!'

        forecaster = Croston(smoothing=smooth)

        fc = forecaster.fit_predict(self.df.loc[:last_train_date], fh=list(range(1, 8)))
        real = self.df.loc[last_train_date + np.timedelta64(1, 'D'):]
        self._save_(real, fc)

    def plot_forecast(self):
        # for testing/debugging
        if self.fc is None:
            print('run forecast first!!')
        else:
            # Plot
            plt.figure(figsize=(12, 5), dpi=100)
            plt.plot(pd.Series(self.real, index=self.index), label='actual')
            plt.plot(pd.Series(self.fc, index=self.index), label='forecast')

            plt.title('Forecast vs Actual')
            plt.legend(loc='upper left', fontsize=8)
            plt.show()


start_time = time.time()
df = pd.read_csv('best_buy_raw_data.csv')
df = df.loc[:, ['Encoded_SKU_ID', 'SALES_DATE', 'DAILY_UNITS']]
df['SALES_DATE'] = pd.to_datetime(df['SALES_DATE'], format='%m/%d/%y')

df2 = pd.read_csv('Validation_Data.csv')
df2 = df2.loc[:, ['Encoded_SKU_ID', 'SALES_DATE', 'DAILY_UNITS']]
df2['SALES_DATE'] = pd.to_datetime(df2['SALES_DATE'], format='%d-%m-%Y')

df = pd.concat((df, df2))

# clean date
df = df.sort_values(by=['Encoded_SKU_ID', 'SALES_DATE'])

SKUs = df['Encoded_SKU_ID'].unique()
DATES = df['SALES_DATE'].unique()
DATES.sort()

# UPDATE THIS
last_train_date = DATES[-8]

temp = auto_forecasts = c_forecasts = arima_forecasts = all_actual = all_skus = np.array([])

start_time = time.time()
for SKU in SKUs:
    # print('runnning for SKU:', SKU)
    Inst = Instance(SKU)  # need to save individual Instances somewhere if reusing!
    Inst.init_df(df)

    # # THIS takes long
    # Inst.ARIMA_forecast((1, 1, 1), (1, 1, 1, 12))
    # arima_forecasts = np.concatenate((arima_forecasts, Inst.fc))

    # # THIS takes too long
    # Inst.AutoARIMA_forecast()
    # auto_forecasts = np.concatenate((auto_forecasts, Inst.fc))

    Inst.Croston_forecast()
    c_forecasts = np.concatenate((c_forecasts, Inst.fc))

    all_actual = np.concatenate((all_actual, Inst.real))
    all_skus = np.concatenate((all_skus, [SKU] * len(Inst.fc)))
    # temp = np.concatenate((temp, [np.mean(Inst.df.to_numpy())] * len(Inst.fc)))

print("--- %s seconds ---" % (time.time() - start_time))

# convert to df
df_final = pd.DataFrame({
    'Dates': np.tile(Inst.index, len(SKUs)),
    'SKUs': all_skus,
    'Forecast': c_forecasts,
    'Actual_Value': all_actual,
})
# RMSE1 = np.sqrt(np.mean((arima_forecasts - all_actual) ** 2))
# RMSE1 = np.sqrt(np.mean((auto_forecasts - all_actual) ** 2))
RMSECro = np.sqrt(np.mean((c_forecasts - all_actual) ** 2))
RMSE0 = np.sqrt(np.mean((np.array([0] * len(all_actual)) - all_actual) ** 2))  # RMSE if predictions are 0
# RMSEAvg = np.sqrt(np.mean((temp - all_actual) ** 2))  # RMSE if predictions are mean
print('RMSE for Croston:', RMSECro)
print('RMSE for all zeros:', RMSE0)
# print('RMSE for prediction = avg sales:', RMSEAvg)
