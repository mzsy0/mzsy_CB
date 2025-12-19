"""
邢不行™️选股框架
Python股票量化投资课程

版权所有 ©️ 邢不行
微信: xbx8662

未经授权，不得复制、修改、或使用本代码的全部或部分内容。仅限个人学习用途，禁止商业用途。

Author: 邢不行
"""
import pandas as pd

# 自定义额外数据
extra_data = {"stock_activation_records": ["时间", "地点", "公司接待人员"]}


# noinspection PyUnusedLocal
def add_factor(df: pd.DataFrame, param=None, **kwargs) -> pd.DataFrame:
    """

    """
    # 从额外参数中获取因子名称
    col_name = kwargs['col_name']

    # 示例因子，仅供解释如何使用外部因子，无实际意义
    factor_col = df['公司接待人员'].apply(lambda x: "董事" in str(x)).astype(int)

    # 创建包含指定因子的DataFrame
    factor_df = pd.DataFrame({col_name: factor_col}, index=df.index)

    return factor_df
