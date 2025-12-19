# app.py
from flask import Flask, render_template, jsonify
import pandas as pd
import requests
import time
from typing import List, Optional
import logging
import re
import threading
import time
import akshare as ak
import numpy as np
from gm.api import *
import os
import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
set_token('c4682ddc445a4a26142043c7bad1dd9683cbf688')  # 输入token

def set_pd_option():
    pd.set_option('display.max_rows', 1000)
    pd.set_option('expand_frame_repr', False)
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)

# --- 你的原始函数 (稍作修改) ---

def get_tx_realtime_quotes(symbols: List[str]) -> Optional[pd.DataFrame]:
    if not symbols:
        return None

    symbol_list = []
    for symbol in symbols:
        if symbol.startswith(('sh', 'sz', 'bj')):
            symbol_list.append(symbol[:8])
        elif symbol.endswith(('SH', 'SZ', 'BJ')):
            prefix = symbol[-2:].lower()
            code = symbol[:6]
            symbol_list.append(f'{prefix}{code}')
        else:
            if symbol.startswith('6'):
                prefix = 'sh'
            elif symbol.startswith(('3', '0')):
                prefix = 'sz'
            elif symbol.startswith('8'):
                prefix = 'bj'
            elif symbol.startswith('11'):
                prefix = 'sh'
            elif symbol.startswith('12'):
                prefix = 'sz'
            else:
                logging.warning(f"无法识别的股票代码格式: {symbol}")
                continue
            symbol_list.append(f'{prefix}{symbol[:6]}')

    if not symbol_list:
        logging.error("没有有效的股票代码")
        return None

    # 修复 URL，移除 'q=' 后的空格
    url = 'https://qt.gtimg.cn/q=' + ','.join(symbol_list)

    try:
        response_text = requests.get(url).text
        pattern = r'v_([^=]+)="([^"]*)"'
        matches = re.findall(pattern, response_text)

        if not matches:
            logging.error("未从响应中找到有效数据")
            return None

        res = []
        for match in matches:
            data_str = match[1]
            s_info = data_str.split('~')

            if len(s_info) < 36:
                logging.warning(f"数据格式不完整，跳过: {match[0]}, 长度: {len(s_info)}")
                continue

            res.append({
                'name': s_info[1],
                'code': s_info[2],
                'price': s_info[3],
                'pre_close': s_info[4],
                'open': s_info[5],
                'high': s_info[33],
                'low': s_info[34],
                'volume': s_info[6],
                'pct_chg': s_info[32],
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
                'amount': s_info[35].split('/')[-1] if '/' in s_info[35] else s_info[35],
                'datetime': s_info[30],
            })

        if not res:
            logging.error("解析后没有有效数据")
            return None

        df = pd.DataFrame(res)

        datetime_series = pd.to_datetime(df['datetime'], format='%Y%m%d%H%M%S', errors='coerce')
        df['date'] = datetime_series.dt.strftime('%Y-%m-%d')
        df['time'] = datetime_series.dt.strftime('%H:%M:%S')
        # 重命名 time 列为 刷新时间
        df = df.rename(columns={'time': '刷新时间'})

        float_cols = ['open', 'pre_close', 'price', 'high', 'low', 'volume', 'pct_chg', 'amount']
        df[float_cols] = df[float_cols].astype(float)

        final_cols = ['name', 'code', 'open', 'pre_close', 'price', 'high', 'low', 'volume', 'pct_chg', 'amount', 'date', '刷新时间']
        df = df[final_cols]

    except requests.exceptions.RequestException as e:
        logging.error(f"获取行情数据失败: {e}")
        return None
    except Exception as e:
        logging.error(f"处理行情数据时发生错误: {e}")
        return None

    return df

def process_stock_data(df: pd.DataFrame, df_local: pd.DataFrame, df_rate: pd.DataFrame, df_bond_name: pd.DataFrame, df_gm: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        logging.error("输入的实时数据为空，无法处理")
        return pd.DataFrame()
    df['name'] = df['name'].str.replace(' ', '', regex=False)
    df_stock = df[~df['code'].str.startswith('1')].copy().rename(columns={'price': '正股价'})
    df_stock = pd.merge(df_stock, df_gm, on='code', how='inner')

    df_bond = df[df['code'].str.startswith('1')].copy().rename(columns={'price': '债现价'})
    df_bond = pd.merge(df_bond, df_bond_name, on='code', how='inner')
    df_bond['name'] = df_bond['正股简称']

    if df_stock.empty or df_bond.empty:
        logging.warning("分离后股票或债券数据为空")
        return pd.DataFrame()

    df_stock['name_prefix'] = df_stock['name'].str[:2]
    df_bond['name_prefix'] = df_bond['name'].str[:2]

    df_merged = pd.merge(
        df_bond[['name_prefix', 'code', '债现价', '刷新时间']], # 包含刷新时间
        df_stock[['name_prefix', '正股价', '股票10分钟成交量', '债券10分钟成交量']],
        on='name_prefix',
        how='inner'
    )


    df_result = pd.merge(df_merged, df_local, on='code', how='inner')

    df_result['股票10分钟最佳成交量'] = (df_result['股票10分钟成交量'] * df_result['转股价']) / 100
    df_result['股票10分钟成交量标记'] = df_result['股票10分钟成交量'].apply(lambda x:f'S-{x/10}')
    df_result['债券10分钟成交量标记'] = df_result['债券10分钟成交量'].apply(lambda x: f'B-{x/10}') 
    df_result['最佳成交量'] = np.where(df_result['债券10分钟成交量'] < df_result['股票10分钟最佳成交量'], df_result['债券10分钟成交量标记'], df_result['股票10分钟成交量标记'])
    # df_stock['最佳成交量'] = df_stock['最佳成交量'].round(decimals=0)
    df_result['股票预设成交量'] = 100000
    df_result['债券预设成交量'] = (df_result['股票预设成交量'] * df_result['转股价']) / 100
    df_result['建议成交时间'] = np.where(df_result['债券10分钟成交量'] < df_result['股票10分钟最佳成交量'], df_result['债券预设成交量'] / df_result['债券10分钟成交量'], df_result['股票预设成交量'] / df_result['股票10分钟成交量'] * 10)
    df_result['建议成交时间'] = (df_result['建议成交时间'] * 10).round(4)

    df_result = df_result[~df_result['转股价'].isna()]
    df_result['转股价值'] = 100 * df_result['正股价'] / df_result['转股价']
    df_result['转股溢价率'] = (df_result['债现价'] / df_result['转股价值']) - 1
    # 注意：百分比格式化在后端处理，前端直接显示
    df_result['转股溢价率_pct'] = df_result['转股溢价率'].map('{:.4%}'.format)

    # 合并阈值数据
    df_result = pd.merge(df_result, df_rate, on='code', how='inner')
    df_result['低于下限'] = ''
    df_result['超过上限'] = ''
    df_result.loc[df_result['转股溢价率'] < df_result['下限'], '低于下限'] = '是'
    df_result.loc[df_result['转股溢价率'] > df_result['上限'], '超过上限'] = '是'

    # 使用 .map 或 .apply 安全填充（避免 SettingWithCopyWarning）
    df_result.loc[:, '股票代码'] = df_result['code'].map(lambda x: bond_to_stock.get(x, {}).get('股票代码'))
    df_result.loc[:, '股票名称'] = df_result['code'].map(lambda x: bond_to_stock.get(x, {}).get('股票名称'))
    df_result.loc[:, '债券名称'] = df_result['code'].map(lambda x: bond_to_stock.get(x, {}).get('债券名称'))

    df_result = df_result.rename(columns={'code':'转债代码'})
    # 只返回前端需要的列
    return df_result[['股票名称', '转债代码', '股票代码', '债现价', '正股价', '转股价', '转股溢价率_pct', '下限', '上限', '低于下限', '超过上限', '最佳成交量', '刷新时间', '建议成交时间']]


# --- Flask 应用设置 ---

app = Flask(__name__)

# --- 初始化和数据缓存逻辑 ---
# 1. 启动时调用 akshare 获取最新转股价
def fetch_and_save_bond_data():
    """使用 akshare 获取转股价并保存"""
    logging.info("开始通过 akshare 获取最新转股价...")
    try:
        df = ak.bond_zh_cov()
        if not df.empty:
            df.to_csv('./data/转股价.csv', encoding='gbk', index=False)
            logging.info("通过 akshare 获取转股价成功并已保存。")
            return True
        else:
            logging.warning("akshare 获取转股价失败，数据为空，沿用上次保存的文件。")
            return False
    except Exception as e:
        logging.error(f"akshare 获取转股价时发生错误: {e}, 沿用上次保存的文件。")
        return False

# 获取股票和债券分钟数据
def download_data(codelist_stock, start_date, end_date) -> pd.DataFrame:

    # 判断时间起始点和需要拼接的部分
    cat = False
    file_path = './data/stock_data.pkl'

    if os.path.exists(file_path):
        local_df = pd.read_pickle(file_path) # 读取本地数据
        if not local_df.empty:
            local_df['bob'] = pd.to_datetime(local_df['bob'], utc=True).dt.tz_convert('Asia/Shanghai')
            df_start_date = local_df['bob'].unique()[0].tz_localize(None)  # 移除时区信息
            df_end_date = local_df['eob'].unique()[-1].tz_localize(None)
            if df_start_date == pd.to_datetime(start_date):
                df_end_date = df_end_date - pd.Timedelta(minutes=1) # 重复获取结束时间前1分钟数据，防止之前请求返回遗漏数据
                start_date = str(df_end_date)
                cat = True
                # print(local_df)
    # print(cat)

    codelist_stock = [i for i in codelist_stock if not i.startswith(('8', '9', '4'))]

    # 拼接，形成掘金终端能接收的股票名称格式
    codelist_stock = ['SHSE.' + code if code[:1] == '6' or code[:2] == '11' else 'SZSE.' + code for code in codelist_stock]

    try:
        df = history(symbol=codelist_stock, frequency='1m', skip_suspended=True,
                               start_time=f'{start_date}',
                               end_time=f'{end_date}', df=True)
    except Exception as e:
        logging.info(f'掘金数据下载失败')

    # 判断是拼接还是重写
    if cat == True:
        df = pd.concat([local_df, df], ignore_index=True)
        df.drop_duplicates(inplace=True)  # 对重复获取的1分钟数据去重
        df.to_pickle('./data/stock_data.pkl')
    else:
        df.to_pickle('./data/stock_data.pkl')


    df = df.groupby('symbol')['volume'].apply(lambda x: x.tail(10).sum()).reset_index(name='10分钟成交量')
    df = df.rename(columns={'symbol':'code'})
    df['code'] = df['code'].str[5:]

    # 构建成交量映射字典（字符串 -> 整数），便于快速查找
    vol_map = dict(zip(df['code'].astype(str), df['10分钟成交量']))
    # 构建结果列表
    results = []

    for stock_code, info in stock_info_dict.items():
        bond_code = info['转债代码']

        # 查找股票和转债的成交量（若缺失，设为无穷大以不影响 min）
        stock_vol = vol_map.get(stock_code, float('inf'))
        bond_vol = vol_map.get(bond_code, float('inf'))

        # 若两者都不存在，跳过（或按需处理）
        if stock_vol == float('inf') and bond_vol == float('inf'):
            continue  # 或 print(f"跳过：{stock_code} 和 {bond_code} 均无成交量数据")

        # 取较小者；若一方缺失，用另一方

        best_vol = min(stock_vol, bond_vol)

        results.append({
            'code': stock_code,
            '股票10分钟成交量': stock_vol,
            '债券10分钟成交量': bond_vol,
        })

    # 生成最终 DataFrame
    result_df = pd.DataFrame(results)

    return result_df


# 执行一次 akshare 数据获取
fetch_and_save_bond_data()

# 2. 读取配置和本地转股价数据
df_rate = pd.read_excel('stock_bond_Summary.xlsx', converters={'股票代码': str, '转债代码': str})
codelist = df_rate['股票代码'].tolist() + df_rate['转债代码'].tolist()

stock_info_dict = {
    str(row['股票代码']): {
        '转债代码': str(row['转债代码']),
        '股票名称': row['股票名称'],
        '债券名称': row['债券名称']
    }
    for _, row in df_rate.iterrows()
}

# 构建反向映射字典：键为转债代码（字符串），值为所需信息
bond_to_stock = {}
for stock_code, info in stock_info_dict.items():
    bond_code = info['转债代码']
    bond_to_stock[bond_code] = {
        '股票代码': stock_code,
        '股票名称': info['股票名称'],
        '债券名称': info['债券名称']
    }

today = pd.Timestamp.today().date()
next_day = today + datetime.timedelta(days=1)
start_date = f'{today} 09:29:00'
end_date = f'{next_day} 09:30:00'


def jj():
    global df_gm
    while True:
        df_gm = download_data(codelist, start_date=start_date, end_date=end_date)
        logging.info('掘金更新成功')
        time.sleep(5)

df_gm = pd.DataFrame()
jj_thread = threading.Thread(target=jj)
jj_thread.daemon = True
jj_thread.start()

# 至少获取一次掘金分钟数据
while True:
    if len(df_gm) != 0:
        break
    time.sleep(1)

codelist = df_rate['股票代码'].tolist() + df_rate['转债代码'].tolist()
df_rate = df_rate.rename(columns={'转债代码': 'code'})
df_rate = df_rate[['code', '下限', '上限']]

df_local = pd.read_csv('./data/转股价.csv', encoding='gbk', converters={'债券代码': str}).rename(columns={'债券代码': 'code'})
df_bond_name = df_local[['code', '正股简称']]
df_local_filtered = df_local[df_local['code'].isin(codelist)][['code', '转股价']]


# 3. 全局变量存储最新数据和警报
latest_data = pd.DataFrame()
latest_alerts = []

def update_data_periodically():
    """后台线程函数，定期更新实时行情数据"""
    global latest_data, latest_alerts
    while True:
        logging.info("开始更新实时行情数据...")
        start_time = time.time()
        df = get_tx_realtime_quotes(codelist)
        if df is not None:
            df_processed = process_stock_data(df, df_local_filtered, df_rate, df_bond_name, df_gm)
            latest_data = df_processed

            # 检查并生成警报，区分类型
            alerts_low = df_processed[df_processed['低于下限'] == '是'].copy()
            alerts_low['alert_type'] = '低于下限'

            alerts_high = df_processed[df_processed['超过上限'] == '是'].copy()
            alerts_high['alert_type'] = '超过上限'

            # 合并两类警报
            alerts_df = pd.concat([alerts_low, alerts_high], ignore_index=True)
            latest_alerts = alerts_df.to_dict(orient='records')

            logging.info(f"实时行情数据更新完成，耗时 {time.time() - start_time:.2f} 秒。当前数据行数: {len(latest_data)}, 警报数量: {len(latest_alerts)}")
        else:
            logging.error("获取实时行情数据失败，跳过本次更新")
            latest_data = pd.DataFrame()
            latest_alerts = []
        time.sleep(1) # 每秒更新一次

# 启动后台更新线程
data_thread = threading.Thread(target=update_data_periodically, daemon=True)
data_thread.start()

# --- Flask 路由 ---

@app.route('/')
def index():
    """提供网页主页"""
    return render_template('index.html')

@app.route('/api/data')
def api_data():
    """提供 JSON 格式的数据 API"""
    # 将最新的数据转换为字典列表返回
    if not latest_data.empty:
        return jsonify(latest_data.to_dict(orient='records'))
    else:
        # 如果缓存数据为空，返回空列表
        return jsonify([])

@app.route('/api/alerts')
def api_alerts():
    """提供警报信息 API"""
    return jsonify(latest_alerts)



if __name__ == '__main__':
    # host='0.0.0.0' 使得服务器可以被局域网内的其他设备访问
    # port=5000 是默认端口，你可以修改
    app.run(host='0.0.0.0', port=5000, debug=False) # debug=False 在生产环境使用