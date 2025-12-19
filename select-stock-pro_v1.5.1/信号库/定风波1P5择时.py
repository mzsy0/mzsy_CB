import pandas as pd

from core.model.strategy_config import StrategyConfig

'''
使用案例
'timing': {
            'name': '定风波1P5择时',  # 择时策略名称
            'limit': 100,
            'factor_list': [ ('开盘至今涨幅', False, None, 1, '0945'),
                            ('隔夜涨跌幅', False, None, 1, '0935')],
            'params': 0.8
        }       
'''


def signal(strategy: StrategyConfig, df_after_limit: pd.DataFrame):
    """
    选股信号
    :param strategy: StrategyConfig, 策略配置
    :param df_after_limit: DataFrame, 数据，包含计算所需要的因子列，包括分钟数据，已经根据股票计算范围进行了精准裁切
    """
    # ======================== 解析策略参数 ===========================
    ratio = strategy.timing.params  # 从配置中解析信号参数
    # 早盘下跌因子
    open_col_name = strategy.timing.factor_list[0].col_name
    # 隔夜下跌因子
    ovn_col_name = strategy.timing.factor_list[1].col_name

    # ======================== 计算下跌比例 ===========================
    # 计算早盘下跌比例
    open_decl_ratio = pd.DataFrame(df_after_limit.groupby('交易日期')[open_col_name].apply(lambda x: (x < 0).mean()))

    # 计算隔夜下跌比例
    ovn_decl_ratio = pd.DataFrame(df_after_limit.groupby('交易日期')[ovn_col_name].apply(lambda x: (x < 0).mean()))

    # 合并早盘和隔夜下跌比例
    decl_ratio = pd.DataFrame(open_decl_ratio[open_col_name] + ovn_decl_ratio[ovn_col_name], columns=['下跌比例'])

    # ======================== 返回择时信号 ===========================
    signals = decl_ratio['下跌比例'].le(ratio).astype(int)  # 下跌比例满足阈值时候，signal为1，否则为0。1表示100%仓位，0表示空仓

    return signals.to_frame(name='择时信号')
