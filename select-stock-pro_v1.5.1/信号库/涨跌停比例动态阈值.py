import pandas as pd

from core.model.strategy_config import StrategyConfig

'''
使用案例
'timing': {
            'name': '涨跌停比例动态阈值',  # 择时策略名称
            'limit': 0,
            'factor_list': [('开盘至今涨幅', False, None, 1, '0945'),
                            ('次日涨跌停状态', False, '涨停', 1, '0945'),
                            ('次日涨跌停状态', False, '跌停', 1, '0945')],
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

    ret_col_name = strategy.timing.factor_list[0].col_name  # 选取第一个因子的名称
    zt_col_name = strategy.timing.factor_list[1].col_name  # 选取第一个因子的名称
    dt_col_name = strategy.timing.factor_list[2].col_name  # 选取第三个因子的名称

    # ======================== 计算下跌比例 ===========================
    zt_num = df_after_limit.groupby('交易日期')[zt_col_name].sum()
    dt_num = df_after_limit.groupby('交易日期')[dt_col_name].sum()
    zdt_ratio = (zt_num) / (dt_num + zt_num + 1e-5).rename('涨跌停比例')
    zdt_ratio = pd.DataFrame(zdt_ratio.rolling(recalls).rank(pct=True).rename('涨跌停比例_rank'))
    decl_ratio = pd.DataFrame(df_after_limit.groupby('交易日期')[ret_col_name].apply(lambda x: (x < 0).mean()))
    decl_ratio.rename(columns={ret_col_name: '下跌比例'}, inplace=True)
    ratio_df = pd.concat([zdt_ratio, decl_ratio], axis=1)

    # ======================== 返回择时信号 ===========================
    signals = (ratio_df['下跌比例'] < ratio_df['涨跌停比例_rank']).astype(int)

    return signals.to_frame(name='择时信号')
