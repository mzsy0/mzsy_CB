from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
from core.model.strategy_config import StrategyConfig

# --------------------------------------------------------------------
# 原策略请查看帖子：https://bbs.quantclass.cn/thread/53041
# --------------------------------------------------------------------
'''
使用案例
'timing': {
            'name': '定风波_长短动量',  # 择时策略名称
            'limit': 0.3,
            'factor_list': [('开盘至今涨幅', False, None, 1, '0945')],
            'params': (5, 60, [0.4, 0.6, 0.6, 0.3], 'sh932000')
        }       
'''


def signal(strategy: StrategyConfig, df_after_limit: pd.DataFrame):
    """
    选股信号
    :param strategy: StrategyConfig, 策略配置
    :param df_after_limit: DataFrame, 数据，包含计算所需要的因子列，包括分钟数据，已经根据股票计算范围进行了精准裁切
    """
    # ======================== 解析策略参数 ===========================
    ret_param, ma_param, param_list, index_code = strategy.timing.params  # 从配置中解析信号参数
    # 因为定风波只用了这样一个因子，所以我们就取用第一个。如果你有多个因子要结合的，自己从timing的对象中找出来就好了。
    col_name = strategy.timing.factor_list[0].col_name  # 选取第一个因子的名称

    # ======================== 计算下跌比例 ===========================
    decl_ratio = pd.DataFrame(df_after_limit.groupby('交易日期')[col_name].apply(lambda x: (x < 0).mean()))
    decl_ratio.rename(columns={col_name: '下跌比例'}, inplace=True)
    ratio_df = load_filter_ratio(ret_param, ma_param, param_list, index_code)
    decl_ratio = pd.concat([decl_ratio, ratio_df], axis=1)

    # ======================== 返回择时信号 ===========================
    signals = (decl_ratio['下跌比例'] < decl_ratio['ratio']).astype(int)  # 返回下跌比例小于ratio的交易日，即为选股信号

    return signals.to_frame(name='择时信号')


def load_filter_ratio(ret_param, ma_param, param_list, index_code):
    index_path = Path(cfg.data_center_path) / f'stock-main-index-data/{index_code}.csv'
    index_df = pd.read_csv(index_path, encoding='gbk', parse_dates=['candle_end_time'])

    index_df = index_df.rename(columns={'close': '收盘价'})
    index_df['指数涨跌幅'] = index_df['收盘价'].pct_change()
    index_df.rename(columns={'candle_end_time': '交易日期'}, inplace=True)
    index_df = index_df[['交易日期', '指数涨跌幅', '收盘价']]

    # 计算长短动量
    ret_factor_name, ma_factor_name = f'Ret_{ret_param}', f'ma_{ma_param}'
    index_df[ret_factor_name] = index_df['收盘价'].pct_change(ret_param)
    index_df[ma_factor_name] = index_df['收盘价'].rolling(window=ma_param).mean()

    con1 = (index_df[ret_factor_name] > 0) & (index_df['收盘价'] > index_df[ma_factor_name])
    con2 = (index_df[ret_factor_name] < 0) & (index_df['收盘价'] > index_df[ma_factor_name])
    con3 = (index_df[ret_factor_name] > 0) & (index_df['收盘价'] < index_df[ma_factor_name])
    con4 = (index_df[ret_factor_name] < 0) & (index_df['收盘价'] < index_df[ma_factor_name])

    # 针对不同的con条件，匹配对应编号的参数param
    param1, param2, param3, param4 = param_list
    index_df['ratio'] = np.select([con1, con2, con3, con4], [param1, param2, param3, param4], default=0.4)

    return index_df[['交易日期', 'ratio']].set_index('交易日期')
