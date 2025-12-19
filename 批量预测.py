#coding=utf8
import warnings
import pandas as pd
import os
from datetime import datetime
import numpy as np
from scipy.stats import norm

def set_pd_option():
    pd.set_option('display.max_rows', 1000)
    pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
    pd.set_option('display.unicode.ambiguous_as_wide', True)  # 设置命令行输出时的列对齐功能
    pd.set_option('display.unicode.east_asian_width', True)

def mix_data(bond_code,period='30m'):

    bond_path_code = bond_code + '.SH' if bond_code[:2] == '11' else bond_code + '.SZ'
    bond_value = pd.read_csv(bond_path + bond_path_code + '.csv', encoding='gbk', skiprows=1)
    bond_value = bond_value[['交易日期','转股价格']]

    # bond_value['交易日期'] = pd.to_datetime(bond_value['交易日期'])
    # bond_value['交易日期'] = bond_value['交易日期'].apply(lambda x:datetime.strftime(x,'%Y-%m-%d'))

    if '/' in bond_value['交易日期'].values[0]:
        bond_value['交易日期'] = bond_value['交易日期'].apply(lambda x:datetime.strptime(x, '%Y/%m/%d').strftime('%Y-%m-%d'))


    stk_code = kzz_basic[kzz_basic['ts_code'].str.startswith(bond_code)]['stk_code'].values[0][2:]
    bond5_df = pd.read_csv(bond_30min_resampled_path+bond_code+'.csv',encoding='gbk',index_col=0)
    bond5_df['交易日期'] = bond5_df['bob'].apply(lambda x:x[:10])
    bond5_df = bond5_df.merge(bond_value, on='交易日期', how='left')

    stk5_df = pd.read_csv(stock_30min_resampled_path + stk_code + '.csv', encoding='gbk', index_col=0)
    # print(bond5_df)
    # print(stk5_df)
    # print(len(bond5_df),len(stk5_df))
    cols = ['交易日期','bob','eob','bond_code',f'转债_{period}_收盘价',f'转债_{period}_成交均价',f'转债_{period}_成交量',f'转债_{period}_成交总金额',
            'stk_code',f'正股_{period}_收盘价',f'正股_{period}_成交均价',f'正股_{period}_成交量',f'正股_{period}_成交总金额','转股价格']
    mix_df = pd.merge(left=bond5_df,right=stk5_df,on='bob',how='left')

    mix_df[f'转债_{period}_收盘价'] = mix_df['close_x']
    mix_df[f'转债_{period}_成交均价'] = mix_df['amount_x'] / mix_df['volume_x']
    mix_df[f'转债_{period}_成交量'] = mix_df['volume_x']
    mix_df[f'转债_{period}_成交总金额'] = mix_df['amount_x']
    mix_df['close_y'] = mix_df['close_y'].fillna(method='bfill')
    mix_df['volume_y'] = mix_df['volume_y'].fillna(value=0)
    mix_df[f'正股_{period}_收盘价'] = mix_df['close_y']
    mix_df[f'正股_{period}_成交均价'] = mix_df['amount_y'] / mix_df['volume_y']
    mix_df[f'正股_{period}_成交量'] = mix_df['volume_y']
    mix_df[f'正股_{period}_成交总金额'] = mix_df['amount_y']
    mix_df['bond_code'] = mix_df['symbol_x'].apply(lambda x:x[-6:])
    mix_df['symbol_y'] = mix_df['symbol_y'].astype(str)
    mix_df['stk_code'] = mix_df['symbol_y'].apply(lambda x: x[-6:])
    mix_df['eob'] = mix_df['eob_x']
    mix_df = mix_df[cols]
    # 转股价填充 有些日期没有转股价 用前一日的填充
    mix_df['转股价格'] = mix_df['转股价格'].fillna(method='ffill')

    # 周期转换
    if period in ['d','D','1d','1D']:

        mix_df['candle_begin_time'] = pd.to_datetime(mix_df['bob'])
        mix_df = mix_df.set_index('candle_begin_time')
        mix_df = mix_df.resample('1D').agg({
            '交易日期':'first',
            'bob':'first',
            'eob':'first',
            'bond_code':'first',
            f'转债_{period}_收盘价':'last',
            f'转债_{period}_成交均价':'mean',
            f'转债_{period}_成交量':'sum',
            f'转债_{period}_成交总金额': 'sum',
            'stk_code':'first',
            f'正股_{period}_收盘价':'last',
            f'正股_{period}_成交均价':'mean',
            f'正股_{period}_成交量':'sum',
            f'正股_{period}_成交总金额':'sum',
            '转股价格':'first'
        })

        mix_df[f'转债_{period}_成交均价'] = mix_df[f'转债_{period}_成交总金额'] / mix_df[f'转债_{period}_成交量']
        mix_df[f'正股_{period}_成交均价'] = mix_df[f'正股_{period}_成交总金额'] / mix_df[f'正股_{period}_成交量']
        mix_df.dropna(inplace=True)
        mix_df.reset_index(inplace=True, drop=False)

        pass


    return mix_df

def add_zt_info(df):
    try:
        stock_code = str(df['stk_code'].values[0])
        zt = pd.read_csv(stock_zt_info_path+stock_code+'.csv',index_col=0,encoding='gbk')
        df = df.merge(zt,on='交易日期')

        return df
    except Exception as e:
        print(e)

def fetch_1m_predict_price(bond_code='',predict_data_date=''):
    stk_code = kzz_basic[kzz_basic['ts_code'].str.startswith(bond_code)]['stk_code'].values[0][2:]
    df = pd.read_csv(stock_1min_path+stk_code+'.csv',index_col=0,encoding='gbk')
    df['交易日期'] = df['bob'].apply(lambda x:x[:10])
    if predict_data_date == '':
        df=df[-238:]
    else:
        df=df[df['交易日期']>=predict_data_date]
    df = df[['bob','close']]
    df.reset_index(inplace=True,drop=True)
    print(df)
    return df
    pass

def predict(**args):

    prepare_N = args['prepare_N']
    predict_M = args['predict_M']
    bond_code = args['bond_code']
    dateRanges = args['dateRanges']
    std_range = args['std_range']
    times = args['times']
    period = args['period']
    removeLimitUpAndDown = args['removeLimitUpAndDown']
    usePredictedClosingPrice = args['usePredictedClosingPrice']
    predict_price = args['predict_price']
    df = args['df']



    df2 = df.copy()
    df2['溢价率'] = round(
        (df2[f'转债_{period}_成交均价'] / (100 / df2['转股价格'] * df2[f'正股_{period}_成交均价']) - 1) * 100, 2)
    df2['转股价值'] = df2[f'正股_{period}_成交均价'] * 100 / df2['转股价格']

    # 限定在固定区间
    if len(dateRanges) != 0:
        con_range = None
        for i in range(len(dateRanges)):
            start_date = dateRanges[i]['start']
            end_date = dateRanges[i]['end']

            con = (df2['交易日期'] >= start_date) & (df2['交易日期'] <= end_date)
            if con_range is None:
                con_range = con
            else:
                con_range = con | con_range

        df2 = df2[con_range]


    last_date = df['交易日期'].values[-1]
    df[f'正股_{period}_波动'] = df[f'正股_{period}_收盘价'].pct_change()
    df[f'正股_{period}_波动_std_{prepare_N}'] = df[f'正股_{period}_波动'].rolling(prepare_N, min_periods=1).std()
    mean = df[f'正股_{period}_收盘价'].rolling(prepare_N).mean().values[-1]
    close = df[f'正股_{period}_收盘价'].values[-1]
    df['pct_mean'] = df[f'正股_{period}_波动'].rolling(prepare_N, min_periods=1).mean()
    pct_mean = df['pct_mean'].values[-1]
    convert_price = df['转股价格'].values[-1]
    sigma_prepare = df[f'正股_{period}_波动_std_{prepare_N}'].values[-1]
    sigma_predict = sigma_prepare * np.sqrt(predict_M)
    pridict_std = sigma_predict

    # 设置预测收盘价
    if usePredictedClosingPrice == 'on':
        predict_close = predict_price
    else:
        predict_close = close
    print('预测用的价格',predict_close)

    # 开始遍历计算
    msg = ''
    begin_std = -1 * times * std_range
    step_std = std_range
    predict_df = pd.DataFrame()  # 标准差概率分布 数学期望表格
    for i in range(0, times * 2):  # times *2 为左右两边
        lower_price = round(predict_close * (1 + (pct_mean + ((begin_std + i * step_std) * sigma_predict))), 3)
        upper_price = round(predict_close * (1 + (pct_mean + ((begin_std + (i + 1) * step_std) * sigma_predict))),
                            3)
        lower_bond_price = round(lower_price * (100 / convert_price), 3)
        upper_bond_price = round(upper_price * (100 / convert_price), 3)

        t_df2 = df2[(df2[f'转股价值'] >= lower_bond_price) &
                    (df2[f'转股价值'] <= upper_bond_price)]
        if removeLimitUpAndDown == 'on':

            his_avg_premium = t_df2[(t_df2[f'正股_{period}_收盘价'] < t_df2['涨停价']) &
                                    (t_df2[f'正股_{period}_收盘价'] > t_df2['跌停价'])]['溢价率'].mean()
        else:
            his_avg_premium = t_df2['溢价率'].mean()

        msg += f'对应的[{round(begin_std + i * step_std, 2)} to {round(begin_std + (i + 1) * step_std, 2)}]个标准差区间为[{lower_price} - {upper_price}]\n 对应的转股价格 [{lower_bond_price} - {upper_bond_price}] <br>'
        if len(dateRanges) != 0:
            msg += f'对应转股价格区间的在{str(dateRanges)} 平均溢价率为{round(his_avg_premium, 2)}<br>'
        else:
            msg += f'对应转股价格区间的历史平均溢价率为{round(his_avg_premium, 2)}<br>'

        # 生成表格
        upper_z_score = round(begin_std + (i + 1) * step_std, 2)
        lower_z_score = round(begin_std + i * step_std, 2)

        upper_cdf_value = round(norm.cdf(upper_z_score), 4)
        lower_cdf_value = round(norm.cdf(lower_z_score), 4)
        _predict_df = pd.DataFrame(
            data=[[f'[{round(begin_std + i * step_std, 2)} {round(begin_std + (i + 1) * step_std, 2)}]',
                   f'[{lower_bond_price} - {upper_bond_price}]',
                   round(upper_cdf_value - lower_cdf_value, 4),
                   round(his_avg_premium, 2)]],
            columns=['标准差区间', '转股价格区间', '概率', '平均溢价'])
        predict_df = pd.concat([predict_df, _predict_df], axis=0)
        # predict_df = predict_df.append(_predict_df)




    print(f'计算过去{prepare_N}个周期的正股波动标准差为{round(sigma_prepare * 100, 2)}%_波动均值为{round(pct_mean * 100, 2)}%<br>')
    print(f'预测的{predict_M}个周期标准差为{round(sigma_predict * 100, 2)}%<br><br>')


    # 输出数学期望
    predict_df.reset_index(drop=True, inplace=True)
    predict_df['range_index'] = predict_df.index + 1
    predict_df['修改平均溢价'] = predict_df['平均溢价']
    up_predict_df = predict_df.iloc[0:times]
    down_predict_df = predict_df.iloc[times:]

    up_predict_df['修改平均溢价'] = up_predict_df['修改平均溢价'].fillna(method='bfill')
    down_predict_df['修改平均溢价'] = down_predict_df['修改平均溢价'].fillna(method='bfill')
    down_predict_df['修改平均溢价'] = down_predict_df['修改平均溢价'].fillna(value=0)
    up_predict_df = pd.concat([up_predict_df, down_predict_df], axis=0)
    # 如果上半区没有值 全部用最近的填充
    up_predict_df['修改平均溢价'] = up_predict_df['修改平均溢价'].fillna(method='bfill')
    # up_predict_df = up_predict_df.append(down_predict_df)
    predict_df = up_predict_df
    del predict_df['range_index']
    predict_df.reset_index(inplace=True, drop=True)

    avg_predict_premium = (predict_df['平均溢价'] * predict_df['概率']).sum()
    predict_df.loc[predict_df['修改平均溢价'] < 0, '修改平均溢价'] = 0
    fix_avg_predict_premium = (predict_df['修改平均溢价'] * predict_df['概率']).sum()

    print('溢价平均数学期望',avg_predict_premium)
    print('修改平均数学期望', fix_avg_predict_premium)
    return avg_predict_premium, fix_avg_predict_premium
    pass


set_pd_option()
warnings.filterwarnings("ignore")
kzz_basic = pd.read_csv('bond_basic.csv',encoding='gbk',index_col=0)
bond_path = 'D:\\Data\\stock_data\\stock-basic-bond\\'
juejin_5m_path = 'D:\\Data\\juejin_5min\\'
stock_5min_path = juejin_5m_path+'stock\\'
bond_5min_path = juejin_5m_path+'bond\\'
juejin_1m_path = 'D:\\Data\\juejin_1min\\'
stock_1min_path = juejin_1m_path+'stock\\'
bond_1min_path = juejin_1m_path+'bond\\'
bond_30min_resampled_path = 'D:\\Data\\juejin_30min_resampled\\'
stock_30min_resampled_path = bond_30min_resampled_path+'stock\\'
bond_30min_resampled_path = bond_30min_resampled_path+'bond\\'
stock_zt_info_path = 'D:\\Data\\stock_data\\stock-zt-info\\'



if __name__ == '__main__':



    end_date = '2025-12-15'
    predict_data_date = '2025-12-16'
    prepare_N = 160
    predict_M = 160
    bond_code = '123230'
    dateRanges = []
    std_range = 0.25
    times = 8
    period = '30min'
    removeLimitUpAndDown = 'off'
    usePredictedClosingPrice = 'on'
    predict_price = 32.94

    predict_prices = fetch_1m_predict_price(bond_code,predict_data_date=predict_data_date)


    df = mix_data(bond_code, period=period)
    df = add_zt_info(df)
    df = df[df['交易日期']<=end_date]


    res = []
    for i in range(len(predict_prices)):
        bob = predict_prices.iloc[i]['bob']
        price = predict_prices.iloc[i]['close']
        args = {
            'prepare_N':prepare_N,
            'predict_M' : predict_M,
            'bond_code' : bond_code,
            'dateRanges' : dateRanges,
            'std_range' : std_range,
            'times' : times,
            'period' : period,
            'removeLimitUpAndDown' : removeLimitUpAndDown,
            'usePredictedClosingPrice' : usePredictedClosingPrice,
            'predict_price':price,
            'df':df
        }
        avg_predict_premium, fix_avg_predict_premium = predict(**args)
        res.append([bob,price,avg_predict_premium, fix_avg_predict_premium])

    df = pd.DataFrame(data=res,columns=['bob','close','溢价平均数学期望','修改溢价平均期望'])
    print(df)
    df.to_csv('result.csv',encoding='gbk')


    pass




