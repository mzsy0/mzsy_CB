from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
from core.model.strategy_config import StrategyConfig

# --------------------------------------------------------------------
# 原策略请查看帖子：https://bbs.quantclass.cn/thread/53433
# --------------------------------------------------------------------
'''
使用案例
'timing': {
            'name': '定风波_macd动态阈值',  # 择时策略名称
            'limit': 0,
            'factor_list': [('开盘至今涨幅', False, None, 1, '0945')],
            'params': (0.5, 0.8, [12, 26, 9], 'sh932000')
        }       
'''


def signal(strategy: StrategyConfig, df_after_limit: pd.DataFrame):
    """
    选股信号
    :param strategy: StrategyConfig, 策略配置
    :param df_after_limit: DataFrame, 数据，包含计算所需要的因子列，包括分钟数据，已经根据股票计算范围进行了精准裁切
    """
    # ======================== 解析策略参数 ===========================
    low_ratio, high_ratio, macd_param_list, index_code = strategy.timing.params  # 从配置中解析信号参数
    # 因为定风波只用了这样一个因子，所以我们就取用第一个。如果你有多个因子要结合的，自己从timing的对象中找出来就好了。
    col_name = strategy.timing.factor_list[0].col_name  # 选取第一个因子的名称

    # ======================== 计算下跌比例 ===========================
    decl_ratio = pd.DataFrame(df_after_limit.groupby('交易日期')[col_name].apply(lambda x: (x < 0).mean()))
    decl_ratio.rename(columns={col_name: '下跌比例'}, inplace=True)
    ratio_df = load_filter_ratio(index_code=index_code, low_ratio=low_ratio, high_ratio=high_ratio, macd_param_list=macd_param_list)
    decl_ratio = pd.concat([decl_ratio, ratio_df], axis=1)

    # ======================== 返回择时信号 ===========================
    signals = (decl_ratio['下跌比例'] < decl_ratio['ratio']).astype(int)  # 返回下跌比例小于ratio的交易日，即为选股信号

    return signals.to_frame(name='择时信号')


def load_filter_ratio(index_code, low_ratio, high_ratio, macd_param_list):
    index_path = Path(cfg.data_center_path) / f'stock-main-index-data/{index_code}.csv'
    index_df = pd.read_csv(index_path, encoding='gbk', parse_dates=['candle_end_time'])

    index_df = index_df.rename(columns={'close': '收盘价'})
    index_df['指数涨跌幅'] = index_df['收盘价'].pct_change()
    index_df.rename(columns={'candle_end_time': '交易日期'}, inplace=True)
    index_df = index_df[['交易日期', '指数涨跌幅', '收盘价']]

    # 计算macd
    # 计算短期与长期 EMA
    fast_period, slow_period, signal_period = macd_param_list
    ema_fast = index_df['收盘价'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = index_df['收盘价'].ewm(span=slow_period, adjust=False).mean()

    # 计算 DIF
    DIF = ema_fast - ema_slow
    # 计算 DEA：对 DIF 进行 EMA 平滑
    DEA = DIF.ewm(span=signal_period, adjust=False).mean()

    index_df['金叉'] = (DIF.shift(1) < DEA.shift(1)) & (DIF >= DEA)
    index_df['死叉'] = (DIF.shift(1) > DEA.shift(1)) & (DIF <= DEA)

    index_df['signal'] = pd.NA
    index_df.loc[index_df['金叉'], 'signal'] = 1
    index_df.loc[index_df['死叉'], 'signal'] = 0
    index_df['signal'] = index_df['signal'].ffill().fillna(1)

    index_df['ratio'] = np.where(index_df['signal'] > 0, high_ratio, low_ratio)

    return index_df[['交易日期', 'ratio']].set_index('交易日期')
