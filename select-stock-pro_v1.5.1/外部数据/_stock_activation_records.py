from pathlib import Path
import numpy as np
import pandas as pd

from config import data_center_path

"""
如何加载自定义数据？
1.准备自定义数据（一个文件夹，里面文件名是日期还是股票代码都行，反正都是通过自己写的【加载自定义数据函数】加载）
2.将自定义数据放在【数据中心】的目录下
3.在本文件中写【加载自定义数据函数】
4.sources字典根据示例中添加【数据名】和【数据路径】（具体规则在sources字典上方的注释中）
"""

ext_data_path = Path(data_center_path) / "stock-activation-records"


def read_ext_data(file_path: str | Path, candle_df: pd.DataFrame, save_cols: list):
    """
    加载股票作为文件名的数据

    示例函数！！！
    示例函数！！！
    示例函数！！！

    :param file_path: 文件路径
    :param candle_df: 行情数据
            股票代码     股票名称     交易日期    开盘价   最高价   最低价   收盘价   前收盘价   成交量 ...
        0   sh600003    S东北高    2007-01-16  4.03    4.03    4.03    4.03    3.84    340624.0
        1   sh600003    S东北高    2007-01-17  4.23    4.23    4.23    4.23    4.03    613290.0
        2   sh600003    S东北高    2007-01-18  4.44    4.44    4.44    4.44    4.23    21280056.0
        3   sh600003    S东北高    2007-01-19  4.44    4.44    4.44    4.44    4.44    0.0
        4   sh600003    S东北高    2007-01-22  4.44    4.44    4.44    4.44    4.44    0.0
    :param save_cols: 需要留下的列
    :return:
    """
    # 个股股票代码
    code = candle_df["股票代码"].iloc[0]
    # 个股数据路径
    path = Path(file_path) / (code + ".csv")
    # 定义最后所需要的列（排除掉candle_df中已有的列）
    new_save_cols = [col for col in save_cols if col not in candle_df.columns]
    if path.exists():
        # 加载自定义数据  此处只是示例，请根据具体数据对相关参数进行修改
        # custom_data = pd.read_csv(path, encoding='gbk', parse_dates=['公告日期'], skiprows=1)
        custom_data = pd.read_csv(
            path, encoding="gbk", parse_dates=["公告日期"], skiprows=1, usecols=["公告日期", "股票代码"] + new_save_cols
        )
        custom_data.rename(columns={"公告日期": "交易日期"}, inplace=True)

        # 根据['交易日期', '股票代码']进行合并，注意！custom_data中必须要包含['交易日期', '股票代码']这两列！名字不同自行修改列名
        candle_df = pd.merge(candle_df, custom_data, on=["交易日期", "股票代码"], how="left")
    else:
        candle_df = candle_df.assign(**{k: np.nan for k in new_save_cols})
    return candle_df
