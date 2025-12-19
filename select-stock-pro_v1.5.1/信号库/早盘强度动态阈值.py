import pandas as pd

from core.model.strategy_config import StrategyConfig

'''
使用案例
'timing': {
            'name': '早盘强度动态阈值',  # 择时策略名称
            'limit': 100,
            'factor_list': [ ('开盘至今涨幅', False, None, 1, '0945')],
            'params': 60
        }       
'''


def signal(strategy: StrategyConfig, df_after_limit: pd.DataFrame):
    """
    选股信号
    :param strategy: StrategyConfig, 策略配置
    :param df_after_limit: DataFrame, 数据，包含计算所需要的因子列，包括分钟数据，已经根据股票计算范围进行了精准裁切
    """
    # ======================== 解析策略参数 ===========================
    recalls = strategy.timing.params  # 从配置中解析信号参数
    # 早盘下跌因子
    col_name = strategy.timing.factor_list[0].col_name

    # ======================== 计算下跌比例 ===========================
    # 计算早盘下跌比例
    decl_ratio = pd.DataFrame(df_after_limit.groupby('交易日期')[col_name].apply(lambda x: (x < 0).mean()))
    decl_ratio.rename(columns={col_name: '下跌比例'}, inplace=True)
    # 计算早盘强度
    ret_mean = pd.DataFrame(df_after_limit.groupby('交易日期')[col_name].apply(lambda x: x.mean()))
    # 计算早盘强度的时序分位数
    ret_mean['时序分位数'] = ret_mean[col_name].rolling(recalls).rank(pct=True)

    # ======================== 返回择时信号 ===========================
    signals = (decl_ratio['下跌比例'] < ret_mean['时序分位数']).astype(int)  # 下跌比例满足阈值时候，signal为1，否则为0。1表示100%仓位，0表示空仓

    return signals.to_frame(name='择时信号')
