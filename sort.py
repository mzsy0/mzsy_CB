import json
import time
import pandas as pd
import requests
from datetime import datetime, timedelta
import os
import urllib3

huatai_path = r'C:\Users\Administrator\Desktop\0829华泰券表查询.xlsx'


def add_market_code(symbol):
    if symbol[:3] == '399':
        return symbol + '.SZ'
    elif symbol[0] == '6':
        return symbol + '.SH'
    elif symbol[0] == '3':
        return symbol + '.SZ'
    elif symbol[0] == '0':
        return symbol + '.SZ'
    elif symbol[:2] in ['SH', 'SZ']:
        return symbol
    elif symbol[0] in ['8', '9', '4']:
        return 'BJ' + symbol
    
# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

tomorrow_search = False
bool_auto_search = True

manual_codeslist_sum_search = False

filter_CB_premium = 7
filter_CB_cValue = 125
filter_increase_rate = 7
filter_decrease_rate = -7

num_of_stock = 1000

date_str = datetime.today().strftime('%Y%m%d')
year = datetime.today().strftime('%Y')
month = datetime.today().strftime('%m')
day = datetime.today().strftime('%d')

if tomorrow_search:
    tomorrow = datetime.today() + timedelta(days=1)

    date_str = tomorrow.strftime('%Y%m%d')
    year = tomorrow.strftime('%Y')
    month = tomorrow.strftime('%m')
    day = tomorrow.strftime('%d')

# filename = f'Data/stock_list/{year}年{int(month)}月{int(day)}日券池.xlsx'
# filename = f'Data/stock_list/2025年8月28日券池.xlsx'

code_file = 'Data/stock_list/HK_Yike_Saineng_DummyMaster_ValuationReport_20250224.xlsx'
code_file = 'Data/input/code.xlsx'

output_dir = f"_output/{date_str}"

def fetch_data():
    def http_request(url, headers={}):
        print(url)
        cookies = {}
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                resp = requests.get(url, headers=headers, cookies=cookies, timeout=30, verify=False)
                return json.loads(resp.text)
            except requests.exceptions.SSLError as e:
                print(f"SSL错误 (尝试 {retry_count + 1}/{max_retries}): {e}")
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(2)
                    continue
                else:
                    print("SSL错误重试次数已达上限，跳过此请求")
                    return None
            except requests.exceptions.RequestException as e:
                print(f"请求错误 (尝试 {retry_count + 1}/{max_retries}): {e}")
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(2)
                    continue
                else:
                    print("请求错误重试次数已达上限，跳过此请求")
                    return None
            except Exception as e:
                print(f"未知错误: {e}")
                return None
        
        return None

    # 新浪财经数据
    sina_url = 'https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeDataSimple?page=1&num=4000&sort=symbol&asc=1&node=hskzz_z&_s_r_a=init'
    list = http_request(sina_url)
    if list is None:
        print("无法获取新浪财经数据，使用空数据")
        list = []
    df = pd.DataFrame(list)

    if df.empty:
        print("新浪财经数据为空，创建空DataFrame")
        df = pd.DataFrame(columns=['code', 'name', 'trade', 'changepercent', 'volume', 'amount', 'ticktime'])

    df = df[['code', 'name', 'trade', 'changepercent', 'volume', 'amount', 'ticktime']]
    rename_dict = {
        'code': '代码',
        'name': '名称',
        'trade': '现价',
        'changepercent': '涨跌幅%',
        'volume': '成交量/手',
        'amount': '成交额/万',
        'ticktime': '更新时间'
    }
    df.rename(columns=rename_dict, inplace=True)

    df[["现价", "涨跌幅%", "成交量/手", "成交额/万"]] = df[["现价", "涨跌幅%", "成交量/手", "成交额/万"]].astype(float)

    df['成交额/万'] *= 0.0001

    df = df[df["名称"].str[2] != '定']

    df = df[df["现价"] > 0]

    df.reset_index(inplace=True, drop=True)

    # 集思录数据
    jsl_url = 'https://www.jisilu.cn/data/cbnew/redeem_list/?___jsl=LST___t={}'.format(int(time.time()) * 1000)
    res = http_request(jsl_url, headers={
        'Connection': 'keep-alive',
        'sec-ch-ua': '^\\^Google',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'X-Requested-With': 'XMLHttpRequest',
        'sec-ch-ua-mobile': '?0',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36',
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'Origin': 'https://www.jisilu.cn',
        'Sec-Fetch-Site': 'same-origin',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Dest': 'empty',
        'Referer': 'https://www.jisilu.cn/data/cbnew/',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    })

    if res is None:
        print("无法获取集思录数据，使用空数据")
        res = {'rows': []}

    data_list = [x['cell'] for x in res['rows']]
    df2 = pd.DataFrame(data_list)

    if df2.empty:
        print("集思录数据为空，创建空DataFrame")
        df2 = pd.DataFrame(columns=['bond_id', 'stock_id', 'stock_nm', 'convert_price', 'curr_iss_amt', 'price', 'sprice',
                                   'redeem_real_days', 'redeem_count_days', 'redeem_total_days', 'redeem_flag', 'redeem_icon', 'delist_dt',
                                   'orig_iss_amt', 'force_redeem'])

    df2 = df2[['bond_id', 'stock_id', 'stock_nm', 'convert_price', 'curr_iss_amt', 'price', 'sprice',
               'redeem_real_days', 'redeem_count_days', 'redeem_total_days', 'redeem_flag', 'redeem_icon', 'delist_dt',
               'orig_iss_amt',
               'force_redeem'
               ]]

    df2['强赎状态'] = df2['redeem_flag'] + df2['redeem_icon']

    df2.loc[df2['强赎状态'] == 'YR', '强赎状态'] = '已公告强赎'
    df2.loc[df2['强赎状态'] == 'XB', '强赎状态'] = '已满足强赎'
    df2.loc[df2['强赎状态'] == 'NG', '强赎状态'] = '公告不强赎'
    df2.loc[df2['强赎状态'] == 'X', '强赎状态'] = ''
    df2.loc[df2['强赎状态'] == 'XR', '强赎状态'] = '公告到期赎回 '
    df2.loc[df2['强赎状态'] == 'YO', '强赎状态'] = '公告提示强赎'

    df2.drop(['redeem_real_days', 'redeem_count_days', 'redeem_total_days',
              'redeem_flag', 'redeem_icon'], axis=1, inplace=True)

    rename_dict = {
        'bond_id': '代码',
        'stock_id': '正股代码',
        'stock_nm': '正股名称',
        'convert_price': '转股价',
        'sprice': "正股价",
        'curr_iss_amt': '剩余规模',
        'price': '现价',
        'delist_dt': "最后交易日",
        'curr_iss_amt': "剩余规模",
        'orig_iss_amt': "发行规模",
        'force_redeem': "强赎信息"
    }
    df2.rename(columns=rename_dict, inplace=True)

    df2[["转股价", "正股价", "剩余规模", "发行规模"]] = df2[["转股价", "正股价", "剩余规模", "发行规模"]].astype(float)

    df2['剩余规模（%）'] = df2['剩余规模'] / df2['发行规模']
    df2['剩余规模（%）'] = df2['剩余规模（%）'].map(lambda x: '{:.2f}%'.format(x * 100))
    df2 = df2[['代码', '正股代码', '正股名称', '正股价', '转股价', '最后交易日', '发行规模', '剩余规模', '剩余规模（%）',
               '强赎状态', '强赎信息']]

    df = df.merge(df2, how='left', on='代码')

    df['转股价值'] = 100 * df["正股价"] / df["转股价"]

    df = df.dropna(subset=['正股代码'])

    df['转股溢价率'] = (df['现价'] / df['转股价值'] - 1) * 100

    df["集思录双低"] = df["现价"] + 100 * df["转股溢价率"]

    df['转股溢价率_得分'] = df['转股溢价率'].rank(ascending=True)
    df['收盘价_得分'] = df['现价'].rank(ascending=True)
    df['剩余规模_得分'] = df['剩余规模'].rank(ascending=True)

    df['多普勒三低'] = df['转股溢价率_得分'] * 2 + df['剩余规模_得分'] * 3 + df['收盘价_得分'] * 2

    df.drop(['转股溢价率_得分', '收盘价_得分', '剩余规模_得分'], axis=1, inplace=True)

    df = df[['代码', '名称', '现价', '涨跌幅%', '正股代码', '正股名称',
             '正股价', '转股溢价率', '转股价', '转股价值', '最后交易日', '发行规模', '剩余规模', '剩余规模（%）',
             '集思录双低', '多普勒三低', '强赎状态', '强赎信息'
             ]]

    df = df.sort_values(by=['多普勒三低'], ascending=True)
    df.reset_index(inplace=True, drop=True)

    return df

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
ind_df = fetch_data()

ind_df['最后交易日'] = ind_df['最后交易日'].fillna('')
ind_df['强赎信息'] = ind_df['强赎信息'].fillna('')
ind_df['强赎信息'].unique()

ind_df_path = os.path.join(output_dir, '市场情况.csv')
ind_df.to_csv(ind_df_path, index=False, encoding='gbk')

DF_filter_ind = ind_df[(ind_df["转股溢价率"] <= filter_CB_premium)
                       | ((ind_df["涨跌幅%"] > filter_increase_rate) | (ind_df["涨跌幅%"] < filter_decrease_rate))
                       ]

stock_code_list = DF_filter_ind["正股代码"].dropna().tolist()

def codetrans(code):
    if len(code) < 6:
        code = (6 - len(code)) * '0' + code + '.SZ'
        return code
    elif len(code) == 6:
        if code[0] in ['6', '5']:
            return code + '.SH'
        else:
            return code + '.SZ'
    else:
        return code

def code_check(code):
    if len(code) < 6:
        code = (6 - len(code)) * '0' + code + '.SZ'
    if len(code) == 6:
        if code[0] in ['6', '5']:
            return code + '.SH'
        elif code[0] in ['8', '9']:
            return code + '.BJ'
        else:
            return code + '.SZ'
    else:
        return code

def get_tx_realtime_quotes(symbols=[]):
    """腾讯接口获取实时行情数据"""
    if not symbols:
        return None
    symbol_list = []
    for symbol in symbols:
        if symbol[:3] == '399':
            symbol_list.append('sz' + symbol)
        elif symbol[0] == '6':
            symbol_list.append('sh' + symbol)
        elif symbol[0] == '3':
            symbol_list.append('sz' + symbol)
        elif symbol[0] == '0':
            symbol_list.append('sz' + symbol)
        elif symbol[:2] in ['sh', 'sz']:
            symbol_list.append(symbol)
        elif symbol[0] in ['8', '9', '4']:
            symbol_list.append('bj' + symbol)

    request = requests.get(url='https://qt.gtimg.cn/q=' + ','.join(symbol_list), verify=False).text.replace('\n', '').split(';')
    request = list(filter(lambda x: len(x) > 0, request))
    res = []
    for s in request:
        s_info = s.split('~')
        res.append({
            'name': s_info[1],
            'code': s_info[2],
            'price': s_info[3],
            'pre_close': s_info[4],
            'open': s_info[5],
            'high': s_info[33],
            'low': s_info[34],
            'volume': s_info[6],
            'b1_p': s_info[9],
            'b1_v': s_info[10],
            'b2_p': s_info[11],
            'b2_v': s_info[12],
            'b3_p': s_info[13],
            'b3_v': s_info[14],
            'b4_p': s_info[15],
            'b4_v': s_info[16],
            'b5_p': s_info[17],
            'b5_v': s_info[18],
            'a1_p': s_info[19],
            'a1_v': s_info[20],
            'a2_p': s_info[21],
            'a2_v': s_info[22],
            'a3_p': s_info[23],
            'a3_v': s_info[24],
            'a4_p': s_info[25],
            'a4_v': s_info[26],
            'a5_p': s_info[27],
            'a5_v': s_info[28],
            'amount': s_info[35].split('/')[2],
            'datetime': s_info[30],
        })

    d = pd.DataFrame(res)
    d['date'] = d['datetime'].apply(lambda x: x[:4] + '-' + x[4:6] + '-' + x[6:8])
    d['time'] = d['datetime'].apply(lambda x: x[-6:-4] + ':' + x[-4:-2] + ':' + x[-2:])

    d = d[['name', 'code', 'open', 'pre_close', 'price', 'high', 'low', 'volume', 'amount', 'b1_v', 'b1_p', 'b2_v',
           'b2_p', 'b3_v', 'b3_p', 'b4_v', 'b4_p', 'b5_v', 'b5_p', 'a1_v', 'a1_p', 'a2_v', 'a2_p', 'a3_v', 'a3_p',
           'a4_v', 'a4_p', 'a5_v', 'a5_p', 'date', 'time']]
    d[['b1_v', 'b1_p', 'b2_v', 'b2_p', 'b3_v', 'b3_p', 'b4_v', 'b4_p', 'b5_v', 'b5_p', 'a1_v', 'a1_p', 'a2_v',
       'a2_p', 'a3_v', 'a3_p', 'a4_v', 'a4_p', 'a5_v', 'a5_p']] = d[
        ['b1_v', 'b1_p', 'b2_v', 'b2_p', 'b3_v', 'b3_p', 'b4_v', 'b4_p', 'b5_v', 'b5_p', 'a1_v', 'a1_p', 'a2_v',
         'a2_p', 'a3_v', 'a3_p', 'a4_v', 'a4_p', 'a5_v', 'a5_p']].astype(float)
    d[['open', 'pre_close', 'price', 'high', 'low', 'volume', 'amount']] = d[
        ['open', 'pre_close', 'price', 'high', 'low', 'volume', 'amount']].astype(float)
    return d

def get_stocknames(codes):
    codes = [code[:6] for code in codes]
    res = []
    for i in range(0, len(codes), 200):
        if i + 200 >= len(codes):
            r = get_tx_realtime_quotes(symbols=codes[i:i + 200])
        else:
            r = get_tx_realtime_quotes(symbols=codes[i:i + 200])
        res.append(r)
    r = pd.concat(res, ignore_index=True)
    r = r.drop_duplicates()
    r.reset_index(drop=True, inplace=True)
    stocknames = {codetrans(code): r[r['code'] == code]['name'].values[0] for code in list(r['code'])}
    stocknames['159949.SZ'] = '创业板50ETF'
    stocknames['511880.SH'] = '银华日利ETF'
    stocknames['512010.SH'] = '医药ETF'
    stocknames['512170.SH'] = '证券保险ETF'
    stocknames['512690.SH'] = '酒ETF'
    return stocknames

if manual_codeslist_sum_search:
    codes_df = pd.read_excel(code_file)
    codes_df['股票代码'] = codes_df['股票代码'].astype(str)
    codes_df['股票代码'] = codes_df['股票代码'].apply(code_check)
    codes_df['股票溢价率顺序'] = codes_df.index
    codes = list(codes_df['股票代码'])

if bool_auto_search:
    codes_df = DF_filter_ind["正股代码"].dropna().astype(str).reset_index(drop=True)
    codes_df = codes_df.to_frame(name="正股代码")
    codes_df['正股代码'] = codes_df['正股代码'].astype(str)
    codes_df['正股代码'] = codes_df['正股代码'].apply(code_check)
    codes_df['股票溢价率顺序'] = codes_df.index
    codes = list(codes_df['正股代码'])

res_df = []
cols = ['券商', '股票代码', '股票名称', '数量', '券息']

# df = pd.read_excel(filename, sheet_name='国君场外')
# df['券商'] = '国君场外'
# df['证券代码'] = df['证券代码'].astype(str)
# df['证券代码'] = df['证券代码'].apply(codetrans)
# df['股票代码'] = df['证券代码']
# df['股票名称'] = df['证券名称']
# df['数量'] = df['可借数量']
# df['券息'] = df['参考券息']
# df = df[cols]

# res_df.append(df)

# df = pd.read_excel(filename, sheet_name='银河德睿')
# df['券商'] = '银河德睿'
# df['标的代码'] = df['标的代码'].astype(str)
# df['标的代码'] = df['标的代码'].apply(codetrans)
# df['股票代码'] = df['标的代码']
# stocknames = get_stocknames(list(df['股票代码']))
# df['股票名称'] = df['标的代码'].apply(lambda x: stocknames[x])
# df['数量'] = df['可出借股数（股）']
# df['券息'] = df['券息']
# df = df[cols]

# res_df.append(df)

# df = pd.read_excel(filename, sheet_name='中证资本')
# df = df[df['召回类型'] == '实时']
# df['券商'] = '中证资本'
# df['股票代码'] = df['标的代码']
# df['股票名称'] = df['标的名称']
# df['数量'] = df['剩余可预约数量']
# df['券息'] = df['费率']
# df = df[cols]
# res_df.append(df)

# df = pd.read_excel(filename, sheet_name='华泰场外')
# df['券商'] = '华泰场外'
# df['股票代码'] = df['标的代码']
# df['股票名称'] = df['标的名称']
# df['数量'] = df['预计可借数量']
# df['券息'] = df['券息率(%)']
# df = df[cols]
# res_df.append(df)

# df = pd.read_excel(filename, sheet_name='建投场外')
# df['券商'] = '建投场外'
# df['股票代码'] = df['标的代码']
# df['股票名称'] = df['标的名称']
# df['数量'] = df['可出借股数']
# df['券息'] = df['票息(%)']
# df = df[cols]
# res_df.append(df)

# df = pd.read_excel(filename, sheet_name='广发场外')
# df['券商'] = '广发场外'
# df['股票代码'] = df['标的代码']
# df['股票名称'] = df['标的名称']
# df['数量'] = df['可预约数量']
# df['券息'] = df['借入利率']
# df = df[cols]
# res_df.append(df)

# df = pd.concat(res_df, ignore_index=True)

# slist_df = df.copy()

# df = df[df['股票代码'].isin(codes)]
# df.reset_index(drop=True, inplace=True)

# os.makedirs(output_dir, exist_ok=True)
# cb_info_path = os.path.join(output_dir, '汇总.csv')
# df.to_csv(cb_info_path, index=False, encoding='gbk')
# filter_slist_df = df.groupby(['股票代码', '券商']).agg({'股票名称': 'sum', '数量': 'sum', '券息': 'sum'})

if manual_codeslist_sum_search:
    codes_only_digits = [c.split('.')[0] for c in codes]
    DF_filter_ind = ind_df[ind_df['正股代码'].isin(codes_only_digits)]

# DF_slist_ind = pd.merge(slist_df.reset_index(), ind_df, left_on='股票名称', right_on='正股名称', how='left')

# DF_slist_ind = DF_slist_ind[['股票代码', '正股代码', '股票名称', '代码', '名称', '转股溢价率',
#                              '券商', '数量', '券息', '正股价', '转股价', '现价', '转股价值', '涨跌幅%', '最后交易日',
#                              '发行规模', '剩余规模', '剩余规模（%）'
#     , '集思录双低', '多普勒三低', '强赎状态', '强赎信息']]
# DF_slist_ind = DF_slist_ind.dropna(subset=['正股代码'])

DF_slist_ind_path = os.path.join(output_dir, '目标约券单｜市场情况.csv')
# DF_slist_ind.to_csv(DF_slist_ind_path, index=False, encoding='gbk')

# DF_filter_slist_ind = pd.merge(slist_df.reset_index(), DF_filter_ind, left_on='股票名称', right_on='正股名称',
#                                how='left')

# DF_filter_slist_ind = DF_filter_slist_ind[['股票代码', '正股代码', '股票名称', '代码', '名称', '转股溢价率',
#                                            '券商', '数量', '券息', '正股价', '转股价', '现价', '转股价值', '涨跌幅%',
#                                            '最后交易日', '发行规模', '剩余规模', '剩余规模（%）'
# #     , '集思录双低', '多普勒三低', '强赎状态', '强赎信息']]
# DF_filter_slist_ind = DF_filter_slist_ind.dropna(subset=['正股代码'])

# DF_slist_ind = DF_slist_ind[DF_slist_ind["数量"] >= num_of_stock]

# DF_filter_slist_ind = DF_filter_slist_ind[DF_filter_slist_ind["数量"] >= num_of_stock]

# DF_filter_slist_ind_output=DF_filter_slist_ind.copy()
columns_to_select = ['股票代码', '股票名称', '券商', '数量', '券息', '转股溢价率', '转股价值', '涨跌幅%', '代码', '名称', '正股价', '转股价', '现价', '最后交易日', '发行规模', '剩余规模', '剩余规模（%）', '集思录双低', '多普勒三低', '强赎状态', '强赎信息']
# DF_filter_slist_ind_output = DF_filter_slist_ind_output[columns_to_select]
# DF_filter_slist_ind_output = DF_filter_slist_ind_output.sort_values(by='股票代码')


# 华泰券单筛选


# df_huatai = pd.read_excel(huatai_path)

# if '预计可借数量' in df_huatai.columns:
#     df_huatai = df_huatai[df_huatai['预计可借数量'] >= 20000]
# else:
#     raise ValueError("没有找到 '预计可借数量' 列，请检查表头！")

# df_huatai['正股代码'] = df_huatai['标的代码'].str[:6]

# DF_filter_ind = DF_filter_ind.copy()

# merged_df = pd.merge(df_huatai, DF_filter_ind, on='正股代码', how='inner', suffixes=('_huatai', '_filter'))

# if '预计可借数量' in merged_df.columns and '正股价' in merged_df.columns:
#     merged_df['市值'] = merged_df['预计可借数量'] * merged_df['正股价']
# else:
#     raise ValueError("没有找到 'INDIC_预计可借数量QTY' 或 '正股价' 列，请检查表头！")

# merged_df = merged_df[merged_df['市值'] >= 500000]

# if 'INDIC_BRW_FEE' in merged_df.columns and '市值' in merged_df.columns:
#     cols = list(merged_df.columns)
#     cols.remove('市值')
#     idx = cols.index('INDIC_BRW_FEE')
#     cols.insert(idx, '市值')
#     merged_df = merged_df[cols]

# merged_df.to_csv('merged_df.csv', index=False, encoding='gbk')
# print(f"merged_df 已导出为 merged_df.csv，共 {len(merged_df)} 行数据")
# print(merged_df)



# 中信里昂券单筛选

import pandas as pd


# date_str = '20250908'
# borrow_path = f'C:\Users\Administrator\Desktop\basket_borrow_pool_{date_str}.xlsx'
# df_borrow = pd.read_excel(borrow_path)

borrow_path = r'C:\Users\Administrator\Desktop\basket_borrow_pool_20251009.xlsx'

df_borrow = pd.read_excel(borrow_path)

# 检查并添加“代码”列
if 'bb_ticker' in df_borrow.columns:
    df_borrow['代码'] = df_borrow['bb_ticker'].str[:6]
else:
    raise ValueError("没有找到 'bb_ticker' 列，请检查表头！")

# 增加筛选条件：只保留 notional CNY 大于等于 500000 的行
if 'notional CNY' in df_borrow.columns:

    df_borrow = df_borrow[df_borrow['notional CNY'] >= 500000]
else:
    raise ValueError("没有找到 'notional CNY' 列，请检查表头！")

# 防止 SettingWithCopyWarning
DF_filter_ind = DF_filter_ind.copy()
DF_filter_ind['正股代码'] = DF_filter_ind['正股代码'].astype(str).str.zfill(6)

# 合并
df_merged = pd.merge(
    df_borrow,
    DF_filter_ind,
    left_on='代码',
    right_on='正股代码',
    how='inner'
)
print(df_merged)
df_merged.to_csv('中信里昂_basket_borrow_pool_20251009.csv',index=False)