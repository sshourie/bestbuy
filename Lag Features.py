import numpy as np
import pandas as pd
# from datetime import timedelta
# from dateutil.relativedelta import relativedelta
# import time
# import importlib

def lagfunc1(df, xx):
    # maybe a better name?
    # number of days on which promotion happened >xx%

    data = []
    for SKU in SKUs:
        df2 = df[df['Encoded_SKU_ID'] == SKU]

        df2 = df2.loc[:, ['SALES_DATE', 'Amt_Discount']].set_index('SALES_DATE')
        df2 = df2.fillna(0)  # this is a necessary step

        qwe = df2.rolling(4 * 7).apply(lambda x: sum(x > xx))
        qwe = qwe.rename(columns={'Amt_Discount': '#days promotion past {} weeks; >{}%'.format(4, xx * 100)})

        # for weeks in [13, 26]:
        #   temp = df2.rolling(weeks*7).apply(lambda x: sum(x>xx))
        #   temp = temp.rename(columns= {'Amt_Discount': '#days promotion past {} weeks; >{}%'.format(weeks, xx*100)})
        #   qwe = qwe.merge(temp, left_index=True, right_index=True)

        qwe['Encoded_SKU_ID'] = SKU
        qwe = qwe.reset_index()
        data.append(qwe)

    data = pd.concat(data)
    return data


def lagfunc2(df):
    # avg price of product (lagged)
    # variable names are bad
    data = []
    for SKU in SKUs:
        df2 = df[df['Encoded_SKU_ID'] == SKU]

        df2 = df2.loc[:, ['SALES_DATE', 'PRICE']].set_index('SALES_DATE')

        qwe = df2.rolling(4 * 7).mean()
        qwe = qwe.rename(columns={'PRICE': 'avg price {} weeks'.format(4)})

        for weeks in [13, 26]:
            temp = df2.rolling(weeks * 7).mean()
            temp = temp.rename(columns={'PRICE': 'avg price {} weeks'.format(weeks)})
            qwe = qwe.merge(temp, left_index=True, right_index=True)

        # AVG price 1 yr ago
        qwe2 = df2.rolling(4 * 52).sum()
        qwe2 = qwe2.rename(columns={'PRICE': 'sum1yr'})

        temp = df2.rolling(4 * 52 + 7 * 1).sum()
        temp = temp.rename(columns={'PRICE': 'sum1yr1w'})
        qwe2 = qwe2.merge(temp, left_index=True, right_index=True)

        temp = df2.rolling(4 * 52 + 7 * 4).sum()
        temp = temp.rename(columns={'PRICE': 'sum1yr4w'})
        qwe2 = qwe2.merge(temp, left_index=True, right_index=True)

        qwe2['avg price 1 weeks, previous year'] = (qwe2['sum1yr1w'] - qwe2['sum1yr']) / 7
        qwe2['avg price 4 weeks, previous year'] = (qwe2['sum1yr4w'] - qwe2['sum1yr']) / 28
        qwe2 = qwe2.loc[:, ['avg price 1 weeks, previous year', 'avg price 4 weeks, previous year']]

        qwe = qwe.merge(qwe2, left_index=True, right_index=True)

        qwe['Encoded_SKU_ID'] = SKU
        qwe = qwe.reset_index()
        data.append(qwe)

    data = pd.concat(data)
    return data


def lagfunc3(df):
    # duplicate of the prev one; add input for func and col name and remove later
    # std dev -> price of product (lagged)
    data = []
    for SKU in SKUs:
        df2 = df[df['Encoded_SKU_ID'] == SKU]

        df2 = df2.loc[:, ['SALES_DATE', 'PRICE']].set_index('SALES_DATE')

        qwe = df2.rolling(4 * 7).std()
        qwe = qwe.rename(columns={'PRICE': 'std dev {} weeks'.format(4)})

        for weeks in [13, 26]:
            temp = df2.rolling(weeks * 7).std()
            temp = temp.rename(columns={'PRICE': 'std dev {} weeks'.format(weeks)})
            qwe = qwe.merge(temp, left_index=True, right_index=True)

        # std 1 yr ago
        qwe2 = df2.rolling(1 * 7).std()
        qwe2 = qwe2.rename(columns={'PRICE': 'std dev 1 weeks, previous year'})

        temp = df2.rolling(4 * 7).std()
        temp = temp.rename(columns={'PRICE': 'std dev 4 weeks, previous year'})

        qwe2 = qwe2.merge(temp, left_index=True, right_index=True)
        qwe2 = qwe2.shift(365)

        qwe = qwe.merge(qwe2, left_index=True, right_index=True)

        qwe['Encoded_SKU_ID'] = SKU
        qwe = qwe.reset_index()
        data.append(qwe)

    data = pd.concat(data)
    return data


def lastpromo(df, xx):
    # number of days since last promo > xx%

    data = all_skus = np.array([])
    for SKU in SKUs:
        df2 = df[df['Encoded_SKU_ID'] == SKU]
        arr = df2['Amt_Discount'].values
        temp = np.where(arr >= xx)
        temp = np.array(temp)
        temp = temp.reshape((temp.shape[1],))
        qwe = temp[1:] - temp[:-1]
        out = np.zeros_like(arr)
        out[temp] = np.concatenate((np.array([0]), qwe))

        data = np.concatenate((data, out))
        all_skus = np.concatenate((all_skus, [SKU] * len(out)))
        # all_dates = np.concatenate((all_dates, df2['SALES_DATE'].values))

    ret = pd.DataFrame({
        'Encoded_SKU_ID': all_skus,
        'time since last promo; >{}%'.format(xx * 100): data,
        'SALES_DATE': df['SALES_DATE'].values,
    })
    return ret


def lastpromo2(df):
    # 2 cols
    # if promo this day
    # if promo this week

    data = df.loc[:, ['Encoded_SKU_ID', 'SALES_DATE', 'Amt_Discount']]
    data['Any promo today'] = data['Amt_Discount'] > 0
    data['Any promo today'] = data['Any promo today'].astype('int')
    data2 = data.loc[:, ['Encoded_SKU_ID', 'SALES_DATE', 'Any promo today']].groupby('Encoded_SKU_ID').rolling(
        7).sum().reset_index()
    data2 = data2.rename(columns={'Any promo today': 'Any promo last week'})
    # data2.columns
    data = data.merge(data2, left_index=True, right_on='level_1')
    data = data.rename(columns={'Encoded_SKU_ID_x': 'Encoded_SKU_ID'})
    data = data.loc[:, ['Encoded_SKU_ID', 'SALES_DATE', 'Any promo today', 'Any promo last week']]

    return data


def full_feat(df, xx):
    SKUs = pd.unique(df['Encoded_SKU_ID'])

    # number of days on which promotion happened >xx%
    data = lagfunc1(df, xx)

    # avg price of product (lagged)
    data = data.merge(lagfunc2(df), on=['SALES_DATE', 'Encoded_SKU_ID'])

    # std dev -> price of product (lagged)
    data = data.merge(lagfunc3(df), on=['SALES_DATE', 'Encoded_SKU_ID'])

    # 7) time since last promotion
    data = data.merge(lastpromo(df, xx), on=['SALES_DATE', 'Encoded_SKU_ID'])

    return data

# df = pd.read_csv('data_imputed.csv')
orig_df = pd.read_csv('final_output_seas-3.csv')
# append validation data here

cols_needed = ['Encoded_SKU_ID',	'SALES_DATE', 'RETAIL_PRICE', 'PROMO_PRICE'	]
df = orig_df.loc[:,cols_needed]

#2) amount of discount %
df['PRICE'] = df['PROMO_PRICE'].fillna(df['RETAIL_PRICE'])
df['Amt_Discount'] = 1-df['PRICE']/df['RETAIL_PRICE']

# sort by SKU and date; ASSUMING that all SKUs have the same date range (taken care of in the data imputation step)
df['SALES_DATE'] = df['SALES_DATE'].astype('datetime64[ns]')
df = df.sort_values(['Encoded_SKU_ID', 'SALES_DATE'])

SKUs = pd.unique(df['Encoded_SKU_ID'])

# avg price of product (lagged)
data = df.merge(lagfunc2(df), on = ['SALES_DATE', 'Encoded_SKU_ID'])

# std dev -> price of product (lagged)
data = data.merge(lagfunc3(df), on = ['SALES_DATE', 'Encoded_SKU_ID'])

#7) time since last promotion
data = data.merge(lastpromo2(df), on= ['SALES_DATE', 'Encoded_SKU_ID'])
