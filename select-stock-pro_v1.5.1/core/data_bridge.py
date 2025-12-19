"""
邢不行｜策略分享会
股票量化策略框架𝓟𝓻𝓸

版权所有 ©️ 邢不行
微信: xbx1717

本代码仅供个人学习使用，未经授权不得复制、修改或用于商业用途。

Author: 邢不行
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config import data_center_path
from core.utils.log_kit import logger

data_center_path = Path(data_center_path)


def auto_load_data(file_path: str | Path, candle_df: pd.DataFrame, save_cols: list):
    """根据数据名，自动加载数据"""
    return _load_normal_data(file_path, candle_df, save_cols)


def _load_normal_data(file_path: str, candle_df: pd.DataFrame, save_cols: list):
    # 个股股票代码
    code = candle_df["股票代码"].iloc[0]
    # 个股分钟数据路径
    path = Path(file_path) / (code + ".csv")
    new_save_cols = [col for col in save_cols if col not in candle_df.columns]
    if path.exists():
        min_data = pd.read_csv(
            path, encoding="gbk", parse_dates=["交易日期"], skiprows=1, usecols=["交易日期", "股票代码"] + new_save_cols
        )
        candle_df = pd.merge(candle_df, min_data, on=["交易日期", "股票代码"], how="left")
    else:
        for col in new_save_cols:
            candle_df[col] = np.nan
    return candle_df


def load_hk_stock(file_path: str, candle_df: pd.DataFrame, save_cols: list) -> pd.DataFrame | None:
    hkd_cny_path = data_center_path / "stock-cny-rate" / "HKD_CNY_rate.csv"
    if not hkd_cny_path.exists():
        logger.error(f"港股数据依赖港元汇率数据：{hkd_cny_path}，请在数据中心订阅“CNY汇率数据”后重试")
        raise FileNotFoundError

    hkd_cny = pd.read_csv(hkd_cny_path, encoding="gbk", skiprows=1, parse_dates=["日期"])
    # 个股股票代码
    code = candle_df["股票代码"].iloc[0]
    # 港股个股数据路径
    hk_stock_path = Path(file_path) / (code + "_HK.csv")
    # 如果可以找到这个港股的个股数据
    if hk_stock_path.exists():
        # 读取港股个股数据
        hk_df = pd.read_csv(
            hk_stock_path,
            encoding="gbk",
            parse_dates=["交易日期"],
            usecols=["交易日期", "收盘价", "前收盘价"],
            skiprows=1,
        )
        hk_df["收盘价"] = hk_df["收盘价"].ffill()
        hk_df["前收盘价"] = hk_df["前收盘价"].ffill()
        # 计算复权因子
        hk_df["复权因子"] = (hk_df["收盘价"] / hk_df["前收盘价"]).cumprod()
        # 计算前复权、后复权收盘价
        hk_df["收盘价_复权"] = hk_df["复权因子"] * (hk_df.iloc[0]["收盘价"] / hk_df.iloc[0]["复权因子"])

        # 合并该股票的A股和港股数据
        temp = pd.merge_ordered(
            hk_df.rename(columns={"交易日期": "交易日期_港股"}),
            candle_df,
            left_on="交易日期_港股",
            right_on="交易日期",
            fill_method="ffill",
            suffixes=("_港股", ""),
        )

        temp.dropna(subset=["交易日期"], inplace=True)
        # 按照交易日期列作为subset，遇到重复的日期，保留最新的数据
        temp = temp.drop_duplicates(subset="交易日期", keep="last")

        # 判断该股票在港股是不是已经退市：如果A股和港股的最新交易日期相差10天以上，就认为该股票已经退市
        if (temp["交易日期"].iloc[-1] - temp["交易日期_港股"].iloc[-1]).days > 10:
            # 获取hk_df最新的交易日期，将data里的收盘价_港股超过这个日期的数据赋值为nan
            last_date = hk_df["交易日期"].iloc[-1]
            temp.loc[temp["交易日期"] > last_date, "收盘价_港股"] = pd.NA

        # 删除港股交易日期列
        temp.drop(columns=["交易日期_港股"], inplace=True)

        # 合并股票数据和汇率数据
        temp = pd.merge_ordered(
            left=temp,
            right=hkd_cny[["日期", "收盘价"]],
            left_on="交易日期",
            right_on="日期",
            fill_method="ffill",
            suffixes=("", "_汇率"),
        )
        temp.dropna(subset=["交易日期"], inplace=True)
        # 按照交易日期列作为subset，遇到重复的日期，保留最新的数据
        temp = temp.drop_duplicates(subset="交易日期", keep="last")
        # 删除汇率交易日期列
        temp.drop(columns=["日期"], inplace=True)

        candle_df = pd.merge(
            candle_df, temp[["交易日期", "收盘价_港股", "收盘价_汇率", "收盘价_复权_港股"]], on="交易日期", how="left"
        )

    # 找不到个股数据，就给个nan值
    else:
        candle_df["收盘价_港股"] = np.nan
        candle_df["收盘价_汇率"] = np.nan
        candle_df["收盘价_复权_港股"] = np.nan
    return candle_df


def load_dividend_delivery(file_path: str, candle_df: pd.DataFrame, save_cols: list):
    # 个股股票代码
    code = candle_df["股票代码"].iloc[0]
    # 个股分红数据路径
    path = Path(file_path) / (code + ".csv")

    keep_cols = [
        "近一年分红",
        "分红率_登记日",
        "分红率_登记日_近年均值",
        "分红率_登记日_近年标准差",
        "分红率_登记日_近年次数",
        "连续分红年份",
    ]

    if path.exists():
        # 读取分红数据
        dividend_data = pd.read_csv(path, encoding="gbk", skiprows=1, parse_dates=["股权登记日", "报告期"])
        # 股权登记日一定是交易日期，为了merge方便，直接重命名
        dividend_data.rename(columns={"股权登记日": "交易日期"}, inplace=True)
        # 删除相同交易日的数据，保留最新的
        dividend_data = dividend_data.drop_duplicates(subset=["交易日期"], keep="last")
        # 把收盘价数据拿过来
        dividend_data = pd.merge(dividend_data, candle_df[["交易日期", "收盘价"]], on="交易日期", how="left")
        # 计算分红率
        dividend_data["分红率_登记日"] = dividend_data["近一年分红"] / dividend_data["收盘价"]
        # 计算登记日的年份
        dividend_data["年份"] = dividend_data["报告期"].dt.year

        # 计算最近3年的分红状态
        for i in dividend_data.index:
            hist_report_date = dividend_data.loc[i, "报告期"] - pd.DateOffset(years=3)
            temp_hist = dividend_data[
                (dividend_data["报告期"] > hist_report_date)
                & (dividend_data["报告期"] <= dividend_data.loc[i, "报告期"])
            ]
            dividend_data.loc[i, "分红率_登记日_近年均值"] = temp_hist["分红率_登记日"].mean()
            dividend_data.loc[i, "分红率_登记日_近年标准差"] = temp_hist["分红率_登记日"].std()
            dividend_data.loc[i, "分红率_登记日_近年次数"] = temp_hist["分红率_登记日"].count()

            # 计算连续多少年分红
            temp_hist = dividend_data[: i + 1].copy()  # 获取至今的所有数据
            dividend_years = list(set(temp_hist["年份"]))  # 有些年份会分好几次
            year_range = list(range(dividend_data.loc[i, "报告期"].year, temp_hist["报告期"].min().year - 1, -1))
            j = 0
            for year in year_range:
                if year in dividend_years:
                    j += 1
                else:
                    break
            dividend_data.loc[i, "连续分红年份"] = j

        # 将分红数据与日线数据和财务数据合并
        temp = pd.merge(
            left=candle_df[["交易日期", "收盘价"]],
            right=dividend_data[["交易日期"] + keep_cols],
            on=["交易日期"],
            how="left",
        )

        # 按照最新交易日期计算分红
        # temp["近一年分红"].fillna(method="ffill", inplace=True)
        temp["近一年分红"] = temp["近一年分红"].ffill()
        temp["分红率_最近日"] = temp["近一年分红"] / temp["收盘价"]
        keep_cols.append("分红率_最近日")

        # ===分红数据只保留270个交易日，如果270日以后还没有分红数据，判定公司下一年不分红了，将分红数据修正为nan
        mark_index = temp[~pd.isnull(temp["分红率_登记日"])].index
        index_list = []
        for index in mark_index:
            index_list += list(range(index, index + 271))
        # index_list可能有重复值，去重
        index_list = list(set(index_list))
        # 先填充，再赋nan
        # temp.fillna(method="ffill", inplace=True)
        temp = temp.ffill()
        # index_list以外的数据，分红数据修正为nan
        temp.loc[~temp.index.isin(index_list), keep_cols] = np.nan

        # 将分红数据与日线数据和财务数据合并
        candle_df = pd.merge(candle_df, temp[["交易日期"] + keep_cols], on="交易日期", how="left")

    else:
        # 没有分红数据，用空值代替
        keep_cols.append("分红率_最近日")
        for col in keep_cols:
            candle_df[col] = np.nan

    return candle_df


def load_15min_data(file_path: str | Path, candle_df: pd.DataFrame, save_cols: list):
    # fmt: off
    save_cols = save_cols or ['0930', '0945', '1000', '1015', '1030', '1045', '1100', '1115', '1130',
                              '1315', '1330', '1345', '1400', '1415', '1430', '1445']
    # fmt: on
    return _load_normal_data(file_path, candle_df, save_cols)


def load_5min_data(file_path: str | Path, candle_df: pd.DataFrame, save_cols: list):
    # fmt: off
    save_cols = save_cols or [
        '0930', '0935', '0940', '0945', '0950', '0955',
        '1000', '1005', '1010', '1015', '1020', '1025', '1030', '1035', '1040', '1045', '1050', '1055',
        '1100', '1105', '1110', '1115', '1120', '1125', '1130',
        '1305', '1310', '1315', '1320', '1325', '1330', '1335', '1340', '1345', '1350', '1355',
        '1400', '1405', '1410', '1415', '1420', '1425', '1430', '1435', '1440', '1445', '1450', '1455'
    ]
    # fmt: on
    return _load_normal_data(file_path, candle_df, save_cols)


def load_stock_notices_title(file_path: str | Path, candle_df: pd.DataFrame, save_cols: list):
    # 个股股票代码
    code = candle_df.at[0, "股票代码"]
    # 数据路径
    path = Path(file_path) / (code + ".csv")
    new_save_cols = [col for col in save_cols if col not in candle_df.columns]
    if path.exists():
        notices_data = pd.read_csv(
            path, encoding="gbk", parse_dates=["公告日期"], skiprows=1, usecols=["公告日期", "股票代码", "公告标题"]
        )
        merge_df = pd.merge_asof(
            notices_data, candle_df[["交易日期"]], left_on="公告日期", right_on="交易日期", direction="backward"
        )
        agg_result = merge_df.groupby("交易日期").agg({"公告标题": lambda x: "==".join(x) if not x.empty else ""})
        agg_result["公告数量"] = merge_df.groupby("交易日期").size()
        candle_df = pd.merge(candle_df, agg_result[["公告标题", "公告数量"]], on="交易日期", how="left").fillna(
            {"公告标题": "", "公告数量": 0}
        )
    else:
        for col in new_save_cols:
            candle_df[col] = "" if col == "公告标题" else np.nan
    return candle_df


presets = {
    # fmt: off
    # 变量命名之后需要统一，-改成_，前两个数据涉及太广，暂时先不改，后续全用_
    # AH港股数据（stock-hk-stock-data），下载地址：https://www.quantclass.cn/data/stock/stock-hk-stock-data
    # 使用案例：extra_data = {'hk-stock': ['收盘价_港股', '收盘价_汇率', '收盘价_复权_港股']}
    "hk-stock": (load_hk_stock, Path(data_center_path) / "stock-hk-stock-data"),

    # 个股分红数据(stock-dividend-delivery)，下载地址：https://www.quantclass.cn/data/stock/stock-dividend-delivery
    # 使用案例：extra_data = {'dividend-delivery': ['近一年分红', '分红率_登记日', '分红率_登记日_近年均值', '分红率_登记日_近年标准差','分红率_登记日_近年次数', '连续分红年份','分红率_最近日']}
    "dividend-delivery": (load_dividend_delivery, Path(data_center_path) / "stock-dividend-delivery"),

    # 股票15分钟收盘价(stock-15m-close-price)，下载地址：https://www.quantclass.cn/data/stock/stock-15m-close-price
    # 使用案例：extra_data = {'15min_close': ['945', '1000', '1015', '1030', '1045', '1100', '1115', '1130', '1315', '1330', '1345', '1400', '1415', '1430', '1445']}
    "15min_close": (load_15min_data, Path(data_center_path) / "stock-15m-close-price"),

    # 股票5分钟收盘价(stock-5m-close-price)，下载地址：https://www.quantclass.cn/data/stock/stock-5m-close-price
    # 使用案例：extra_data = {'5min_close': ['935', '940', '945', '950', '955', '1000', '1005', '1010', '1015', '1020', '1025', '1030', '1035', '1040', '1045', '1050', '1055', '1100', '1105', '1110', '1115', '1120', '1125', '1130', '1305', '1310', '1315', '1320', '1325', '1330', '1335', '1340', '1345', '1350', '1355', '1400', '1405', '1410', '1415', '1420', '1425', '1430', '1435', '1440', '1445', '1450', '1455']}
    "5min_close": (load_5min_data, Path(data_center_path) / "stock-5m-close-price"),

    # 筹码分布市场数据(stock-chip-distribution)，下载地址：https://www.quantclass.cn/data/stock/stock-chip-distribution
    # 使用案例：extra_data = {'stock_chip_distribution': ['后复权价格', '历史最低价', '历史最高价', '5分位成本', '10分位成本', '15分位成本', '20分位成本', '25分位成本', '30分位成本', '35分位成本', '40分位成本', '45分位成本', '50分位成本', '55分位成本', '60分位成本', '65分位成本', '70分位成本', '75分位成本', '80分位成本', '85分位成本', '90分位成本', '95分位成本', '加权平均成本', '胜率']}
    "stock_chip_distribution": (auto_load_data, Path(data_center_path) / "stock-chip-distribution"),

    # 摆动指标因子(stock-oscillator-factors)，下载地址：https://www.quantclass.cn/data/stock/stock-oscillator-factors
    # 使用案例：extra_data = {'stock_oscillator_factors': ['coppock', 'coppock_5_衰减加权', 'coppock_20_衰减加权', 'SRMi', 'SRMi_5_衰减加权', 'SRMi_20_衰减加权']}
    "stock_oscillator_factors": (auto_load_data, Path(data_center_path) / "stock-oscillator-factors"),

    # 技术指标因子(stock-technical-factors)，下载地址：https://www.quantclass.cn/data/stock/stock-technical-factors
    # 使用案例：extra_data = {'stock_technical_factors': ['ATR', 'ATR_5_衰减加权', 'ATR_20_衰减加权']}
    "stock_technical_factors": (auto_load_data, Path(data_center_path) / "stock-technical-factors"),

    # 反趋向指标因子(stock-anti-trend-factors)，下载地址：https://www.quantclass.cn/data/stock/stock-anti-trend-factors
    # 使用案例：extra_data = {'stock_anti_trend_factors': ['Bias_min', 'Bias_min_5_衰减加权', 'Bias_min_20_衰减加权', 'CCI', 'CCI_5_衰减加权', 'CCI_20_衰减加权', 'RSI', 'RSI_5_衰减加权', 'RSI_20_衰减加权']}
    "stock_anti_trend_factors": (auto_load_data, Path(data_center_path) / "stock-anti-trend-factors"),

    # 量价指标因子(stock-volume-price-factors)，下载地址：https://www.quantclass.cn/data/stock/stock-volume-price-factors
    # 使用案例：extra_data = {'stock_volume_price_factors': ['EOM', 'EOM_5_衰减加权', 'EOM_20_衰减加权', 'Money_Flow', 'Money_Flow_5_衰减加权', 'Money_Flow_20_衰减加权', 'PVT', 'PVT_5_衰减加权', 'PVT_20_衰减加权']}
    "stock_volume_price_factors": (auto_load_data, Path(data_center_path) / "stock-volume-price-factors"),

    # 能量指标因子(stock-energy-factors)，下载地址：https://www.quantclass.cn/data/stock/stock-energy-factors
    # 使用案例：extra_data = {'stock_energy_factors': ['VR成交量比率', 'VR成交量比率_5_衰减加权', 'VR成交量比率_20_衰减加权', '人气指标BR', '人气指标BR_5_衰减加权', '人气指标BR_20_衰减加权', '中间意愿指标CR', '中间意愿指标CR_5_衰减加权', '中间意愿指标CR_20_衰减加权']}
    "stock_energy_factors": (auto_load_data, Path(data_center_path) / "stock-energy-factors"),

    # 趋向指标因子(stock-trend-factors)，下载地址：https://www.quantclass.cn/data/stock/stock-trend-factors
    # 使用案例：extra_data = {'stock_trend_factors': ['MACD', 'MACD_5_衰减加权', 'MACD_20_衰减加权', 'MTM_ma', 'MTM_ma_5_衰减加权', 'MTM_ma_20_衰减加权', '收集派发_ACD', '收集派发_ACD_5_衰减加权', '收集派发_ACD_20_衰减加权']}
    "stock_trend_factors": (auto_load_data, Path(data_center_path) / "stock-trend-factors"),

    # 多因子系列(stock-multi-factor-series)，下载地址：https://www.quantclass.cn/data/stock/stock-multi-factor-series
    # 使用案例：extra_data = {'stock_multi_factor_series': ['潮汐因子_强势半潮汐', '适度冒险', '适度冒险_月耀眼波动率', '勇攀高峰_月稳攀登', '云开雾散_月均模糊关联度', '云开雾散_模糊关联度']}
    "stock_multi_factor_series": (auto_load_data, Path(data_center_path) / "stock-multi-factor-series"),

    # 股票公告标题汇总(stock-notices-title)，下载地址：https://www.quantclass.cn/data/stock/stock-notices-title
    # 使用案例：extra_data = {'stock_notices_title': ['公告标题','公告数量']}
    "stock_notices_title": (load_stock_notices_title, Path(data_center_path) / "stock-notices-title"),
    # fmt: on
}
