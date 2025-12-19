# -*- coding: utf-8 -*-
"""
邢不行｜策略分享会
股票量化策略框架𝓟𝓻𝓸

版权所有 ©️ 邢不行
微信: xbx1717

本代码仅供个人学习使用，未经授权不得复制、修改或用于商业用途。

Author: 邢不行
"""

import numpy as np
import pandas as pd
from pathlib import Path
import tools.utils.tfunctions as tf
import tools.utils.pfunctions as pf
from core.model.backtest_config import load_config
import os
import warnings
from urllib.parse import quote

warnings.filterwarnings("ignore")


# region ===== 辅助函数 =====
# 读取选股结果数据
def load_select_data(_strategy_name, _start_time, _end_time, _results_dir) -> pd.DataFrame:
    """加载策略选股结果"""
    file_path = os.path.join(_results_dir, f"选股结果.pkl")
    temp = pd.read_pickle(file_path)
    if temp.empty:
        raise ValueError(f"{_strategy_name} 选股结果文件为空，请检查数据")
    df = temp[temp["策略"] == _strategy_name]
    # 类型转换
    str_cols = ["股票代码", "股票名称", "策略", "持仓周期", "换仓时间"]
    df[str_cols] = df[str_cols].astype(str)
    # 时间过滤
    df = df[(df["选股日期"] >= pd.to_datetime(_start_time)) & (df["选股日期"] <= pd.to_datetime(_end_time))]
    if df.empty:
        raise ValueError(f"回测时间和分析时间没有交集，请检查数据")
    return df


# 读取k线数据
def load_kline_data(_stocks, _all_add_factor, _cache_dir) -> pd.DataFrame:
    all_factors_kline = pd.read_pickle(os.path.join(_cache_dir, "all_factors_kline.pkl"))
    if all_factors_kline.empty:
        raise ValueError("回测文件夹下 all_factors_kline 文件数据为空，请检查数据")
    all_factors_kline = all_factors_kline[["交易日期", "股票代码"]]

    # 读取所需因子信息
    factors_pkl = [_dir[7:-4] for _dir in os.listdir(_cache_dir) if _dir.startswith("factor_")]
    for factor_name in _all_add_factor:
        if factor_name not in factors_pkl + ["指数"]:
            raise ValueError(f"{factor_name} 因子不存在，请检查数据")
        if factor_name in factors_pkl:
            factor = pd.read_pickle(os.path.join(_cache_dir, f"factor_{factor_name}.pkl"))
            if factor.empty:
                raise ValueError(f"{factor} 因子数据为空，请检查数据")
            if len(all_factors_kline) != len(factor):
                raise ValueError(f"{factor} 因子长度不匹配，需要重新回测，更新数据")
            all_factors_kline[factor_name] = factor

    # 只保留所需股票的数据
    all_factors_kline = all_factors_kline[all_factors_kline["股票代码"].isin(_stocks)]
    # 调整数据格式
    all_factors_kline[["股票代码"]] = all_factors_kline[["股票代码"]].astype(str)
    all_factors_kline = all_factors_kline.sort_values(by=["股票代码", "交易日期"])

    # 读取全部的股票行情数据
    stocks_data_dict = pd.read_pickle(os.path.join(_cache_dir, "股票预处理数据.pkl"))
    if not stocks_data_dict:
        raise ValueError("回测文件夹下 股票预处理数据 文件数据为空，请检查数据")
    all_data = pd.concat(stocks_data_dict.values())
    del stocks_data_dict
    # 与因子数据合并
    all_factors_kline = pd.merge(all_factors_kline, all_data, on=["交易日期", "股票代码"], how="left")
    return all_factors_kline


# 初始化目录
def init_directories(_strategy_name, _backtest_name, _start_time, _end_time, _analysis_dir) -> tuple:
    """初始化结果保存目录"""
    _save_path = os.path.join(
        _analysis_dir, f"{_backtest_name}/{str(_strategy_name).replace('#', '')}_{_start_time}_{_end_time}/"
    )
    _fig_save_path = os.path.join(_save_path, "选股行情图/")
    os.makedirs(_save_path, exist_ok=True)
    os.makedirs(_fig_save_path, exist_ok=True)
    return _save_path, _fig_save_path


# offset数据
def merge_period_offset(_select: pd.DataFrame, _period_offset_df: pd.DataFrame) -> pd.DataFrame:
    result_df = pd.DataFrame()
    for period_offset in _select["持仓周期"].unique():
        _sing_period_offset_select = _select[_select["持仓周期"] == period_offset].copy()
        # offset信息
        result_df_temp = _period_offset_df[["交易日期"]].copy()
        result_df_temp["持有开始"] = _period_offset_df["交易日期"].copy()
        result_df_temp["持有到期"] = _period_offset_df["交易日期"].copy()
        result_df_temp["_group"] = _period_offset_df[period_offset].copy()
        result_df_temp.loc[result_df_temp["_group"] < 0, "持有到期"] = None
        result_df_temp[f"持有天数"] = 1
        result_df_temp.loc[result_df_temp["_group"] < 0, "持有天数"] = 0
        result_df_temp["group"] = result_df_temp["_group"].abs()

        po_df = (
            result_df_temp.groupby([f"group"])
            .agg({"持有开始": "first", "持有到期": "last", "交易日期": "last", "持有天数": "sum"})
            .reset_index()
        )
        po_df["持有周期"] = po_df["持有开始"].dt.date.apply(str) + "--" + po_df["持有到期"].dt.date.apply(str)

        po_df["持有开始"] = po_df["持有开始"].shift(-1)
        po_df["持有到期"] = po_df["持有到期"].shift(-1)
        po_df["持有周期"] = po_df["持有周期"].shift(-1)
        po_df["持有天数"] = po_df["持有天数"].shift(-1)
        po_df.rename(columns={"交易日期": "选股日期"}, inplace=True)
        _sing_period_offset_select = pd.merge(
            _sing_period_offset_select,
            po_df[["选股日期", "持有开始", "持有到期", "持有周期", "持有天数"]],
            on="选股日期",
            how="left",
        )
        result_df = pd.concat([result_df, _sing_period_offset_select], ignore_index=True)
    return result_df


# 计算复权价格
def calculate_adjusted_prices(price_df: pd.DataFrame, _rebalanced_time: str) -> pd.DataFrame:
    """
    计算复权价格
    """
    # 计算分钟价格的复权价格
    if _rebalanced_time not in ["close-open", "close", "open"]:
        _rebalanced_time_5min = _rebalanced_time.split("-")[0]
        price_df[f"{_rebalanced_time_5min}_复权"] = (
            price_df[_rebalanced_time_5min] / price_df["收盘价"] * price_df["收盘价_复权"]
        )
    else:
        return price_df
    # 对于 close-open / open / close 三种模式，直接返回数据
    return price_df


# 计算持有期间的收益率
def get_buy_sell_ret(_all_factors_kline, _select, _rebalanced_time, _c_rate, _t_rate):
    """
    用复权价格计算股票持仓时间内的整体收益率
    """
    # 预处理数据，生成需要的列
    if _rebalanced_time == "close":
        _all_factors_kline[["上日_收盘价", "上日_收盘价_复权"]] = _all_factors_kline.groupby("股票代码")[
            ["收盘价", "收盘价_复权"]
        ].shift(1)
    elif _rebalanced_time == "open":
        _all_factors_kline[["下日_开盘价", "下日_开盘价_复权"]] = _all_factors_kline.groupby("股票代码")[
            ["开盘价", "开盘价_复权"]
        ].shift(-1)
    elif _rebalanced_time not in ["close", "open", "close-open"]:
        _rebalanced_time_5min = _rebalanced_time.split("-")[0]
        _all_factors_kline[[f"下日_{_rebalanced_time_5min}", f"下日_{_rebalanced_time_5min}_复权"]] = (
            _all_factors_kline.groupby("股票代码")[[f"{_rebalanced_time_5min}", f"{_rebalanced_time_5min}_复权"]].shift(
                -1
            )
        )

    # 确定合并的列和日期字段
    if _rebalanced_time == "close-open":
        buy_cols, sell_cols = ["开盘价", "开盘价_复权"], ["收盘价", "收盘价_复权"]
        buy_date, sell_date = "持有开始", "持有到期"
    elif _rebalanced_time == "close":
        buy_cols, sell_cols = ["上日_收盘价", "上日_收盘价_复权"], ["收盘价", "收盘价_复权"]
        buy_date, sell_date = "持有开始", "持有到期"
    elif _rebalanced_time == "open":
        buy_cols, sell_cols = ["开盘价", "开盘价_复权"], ["下日_开盘价", "下日_开盘价_复权"]
        buy_date, sell_date = "持有开始", "持有到期"
    else:
        _rebalanced_time_5min = _rebalanced_time.split("-")[0]
        buy_cols = [f"{_rebalanced_time_5min}", f"{_rebalanced_time_5min}_复权"]
        sell_cols = [f"下日_{_rebalanced_time_5min}", f"下日_{_rebalanced_time_5min}_复权"]
        buy_date, sell_date = "持有开始", "持有到期"

    # 分别合并买入价格和卖出价格
    _select = pd.merge(
        _select,
        _all_factors_kline[["交易日期", "股票代码"] + buy_cols],
        left_on=[buy_date, "股票代码"],
        right_on=["交易日期", "股票代码"],
        how="left",
    ).drop(columns=["交易日期"])

    _select = pd.merge(
        _select,
        _all_factors_kline[["交易日期", "股票代码"] + sell_cols],
        left_on=[sell_date, "股票代码"],
        right_on=["交易日期", "股票代码"],
        how="left",
    ).drop(columns=["交易日期"])

    # 计算收益率
    _select["持有周期收益率"] = (_select[sell_cols[1]] / _select[buy_cols[1]] - 1) * (1 - _c_rate * 2 - _t_rate)
    _select["持有周期收益率"] = _select["持有周期收益率"].round(4)

    return _select


# 交易表§
def get_trade_info(_select, _rebalanced_time):
    """
    整理交易信息，用于html展示，重点在与买入价格和卖出价格的确认
    """
    df = _select.copy()
    common_map = {"持有开始": "买入日期", "持有到期": "卖出日期", "持有周期收益率": "收益率"}
    # 买入价格和卖出价格是非复权价格
    specific_map = {
        "close-open": {"开盘价": "买入价", "收盘价": "卖出价"},
        "open": {"开盘价": "买入价", "下日_开盘价": "卖出价"},
        "close": {"上日_收盘价": "买入价", "收盘价": "卖出价"},
    }

    if _rebalanced_time in specific_map:
        mapping = {**common_map, **specific_map[_rebalanced_time]}
    else:
        time_key = _rebalanced_time.split("-")[0]
        mapping = {**common_map, time_key: "买入价", f"下日_{time_key}": "卖出价"}

    df.rename(columns=mapping, inplace=True)
    df["_持仓周期"] = df["持仓周期"].apply(lambda x: x.replace("_", ""))
    df = df.sort_values(["股票代码", "买入日期"])
    return df[["股票代码", "_持仓周期", "持仓周期", "买入日期", "卖出日期", "买入价", "卖出价", "收益率"]]


# 检查单个股票是否存在绘图所需因子数据
def check_factor_in_df(single_stock_df, main_factor_list, sub_factor_list):
    main_factor_list_filter = []
    sub_factor_list_filter = []

    err_factor = []
    for each_factor in main_factor_list:
        if each_factor["因子名称"] == "指数":
            main_factor_list_filter.append(each_factor)
        elif each_factor["因子名称"] in single_stock_df.columns:
            main_factor_list_filter.append(each_factor)
        else:
            err_factor.append(each_factor["因子名称"])
    for item in sub_factor_list:
        factor_names = item["因子名称"]
        common_factors = []
        for factor in factor_names:
            if (factor in single_stock_df.columns) or (factor == "指数"):
                common_factors.append(factor)
            else:
                err_factor.append(factor)
        if common_factors:
            sub_factor_list_filter.append({"因子名称": common_factors, "图形样式": item["图形样式"]})
    err_factor = list(set(err_factor))
    if len(err_factor):
        print(f'{"、".join(err_factor)} 因子不存在')
    return main_factor_list_filter, sub_factor_list_filter


# 汇总股票交易信息
def analyze_stock_selection(_select: pd.DataFrame):
    """
    对股票数据进行分组分析，返回各股票的统计结果及总体描述性统计数据。
    """
    res_list = []  # 存储每个分组的结果

    # 遍历每个股票分组
    for stock_name, group in _select.groupby(["股票代码"]):
        # 对每个分组按照选股日期排序
        group.sort_values(by="选股日期", inplace=True)

        # 初始化结果临时DataFrame
        res_temp = pd.DataFrame()
        res_temp.loc[0, "股票代码"] = stock_name[0]
        res_temp.loc[0, "股票名称"] = group["股票名称"].iloc[-1]
        res_temp.loc[0, "选中次数"] = len(group["选股日期"].unique())
        res_temp.loc[0, "累计持股天数"] = group["持有天数"].sum()
        offset_ret = []
        for offset in group["持仓周期"].unique():
            offset_temp = group[group["持仓周期"] == offset]
            offset_temp = offset_temp.sort_values(by=["选股日期"])
            offset_ret.append((offset_temp["持有周期收益率"] + 1).prod() - 1)
        res_temp.loc[0, "累计持股收益"] = np.mean(offset_ret)
        res_temp.loc[0, "次均收益率"] = group["持有周期收益率"].mean()
        res_temp.loc[0, "首次选中时间"] = group["选股日期"].dt.date.iloc[0]
        res_temp.loc[0, "最后选中时间"] = group["选股日期"].dt.date.iloc[-1]

        # 插入持有周期列表
        res_temp["持有周期"] = ""  # 赋值一个空字符串确保列为object类型
        res_temp.at[0, "持有周期"] = group["持有周期"].to_list()

        # 将当前分组结果添加到结果列表
        res_list.append(res_temp)

    # 汇总所有分组的分析结果
    all_res = pd.concat(res_list, ignore_index=True)

    # 对总体数据进行描述性统计
    describe = pd.DataFrame()
    describe.loc[0, "选股数"] = all_res.shape[0]
    describe.loc[0, "平均选中次数"] = all_res["选中次数"].mean()
    describe.loc[0, "平均累计持股天数"] = all_res["累计持股天数"].mean()
    describe.loc[0, "平均次均收益率"] = all_res["次均收益率"].mean()
    describe.loc[0, "平均持股累计收益"] = all_res["累计持股收益"].mean()
    describe.loc[0, "选股胜率"] = all_res[all_res["累计持股收益"] > 0].shape[0] / describe.loc[0, "选股数"]

    return all_res, describe.T


# endregion


# ===== 策略查看器主函数 =====
def main(_config):
    # 获取配置信息
    _backtest_name = config["backtest_name"]
    _start_time = _config["start_time"]
    _end_time = _config["end_time"]
    _add_days = _config["add_days"]
    _strategy_name_temp = _config["strategy_name"]
    _add_factor_main_list = _config["add_factor_main_list"]
    _add_factor_sub_list = _config["add_factor_sub_list"]
    _color_dict = _config["color_dict"]
    config_global = load_config()

    # 统一处理，将因子名改为不要以factor_开头的因子名
    for v in _add_factor_main_list:
        if v["因子名称"].startswith("factor_"):
            v["因子名称"] = v["因子名称"][7:]
    for v in _add_factor_sub_list:
        v["因子名称"] = [x[7:] if x.startswith("factor_") else x for x in v["因子名称"]]

    strategy_names_list = [strategy.name for strategy in config_global.strategy_list]

    # 检查并规范策略名字
    # 第一种情况：输入数字，策略位置
    if isinstance(_strategy_name_temp, (int, float)):
        if _strategy_name_temp >= len(config_global.strategy_list) or _strategy_name_temp < 0:
            raise ValueError(f"{_strategy_name_temp} 数字输入不符合策略范围，请检查")
        _strategy_name = config_global.strategy_list[_strategy_name_temp].name

    # 第二种情况，输入字符串
    elif isinstance(_strategy_name_temp, str):
        # 如果字符串是策略name，且考虑重名情况，若重名则取第一个策略
        if _strategy_name_temp in config_global.strategy_name_list:
            _strategy_name = next((x for x in strategy_names_list if _strategy_name_temp == x.split(".")[1]), None)
        # 策略按照规范名称输入
        elif _strategy_name_temp in strategy_names_list:
            _strategy_name = _strategy_name_temp
        else:
            raise ValueError(f"{_strategy_name_temp}名称输入有误，请检查")
    else:
        raise ValueError(f"{_strategy_name_temp} 名称输入未按照规定的三种方式输入，请检查")

    # 加载config设置
    c_rate = config_global.c_rate  # 手续费
    t_rate = config_global.t_rate  # 印花税
    data_center_path = config_global.data_center_path  # 数据中心路径 # 回测名称
    root_dir = config_global.get_result_folder().parent.parent  # 根目录
    results_dir = config_global.get_result_folder().parent / _backtest_name
    analysis_dir = os.path.join(root_dir, "分析结果/策略查看器")  # 分析结果保存目录
    cache_dir = os.path.join(root_dir, "运行缓存")  # 策略数据保存路径

    # 初始化结果目录
    save_path, fig_save_path = init_directories(_strategy_name, _backtest_name, _start_time, _end_time, analysis_dir)

    # 整合除K线外的绘图数据
    all_add_factor = [item["因子名称"] for item in _add_factor_main_list] + [
        name for item in _add_factor_sub_list for name in item["因子名称"]
    ]
    all_add_factor = list(set(all_add_factor))
    # 如果是 factor_ 开头的，则只保留后半部分
    all_add_factor = [factor[7:] if factor.startswith("factor_") else factor for factor in all_add_factor]

    # K线开始时间
    d_start = pd.to_datetime(_start_time) - pd.to_timedelta(f"{_add_days}d")  # 日线数据开始时间
    # K线结束时间
    d_end = pd.to_datetime(_end_time) + pd.to_timedelta(f"{_add_days}d")  # 日线数据结束时间

    # 初始化数据
    select = load_select_data(_strategy_name, _start_time, _end_time, results_dir)
    stocks = list(select["股票代码"].unique())
    all_factors_kline = load_kline_data(stocks, all_add_factor, cache_dir)
    period_offset_df = pd.read_csv(
        Path(data_center_path) / "period_offset.csv", encoding="gbk", skiprows=1, parse_dates=["交易日期"]
    )
    index_data = tf.import_index_data(os.path.join(config_global.index_data_path, "sh000001.csv"), (d_start, d_end))

    # 策略换仓时间
    rebalanced_time = select["换仓时间"].unique()[0]

    # 计算分钟复权价格
    all_factors_kline = calculate_adjusted_prices(all_factors_kline, rebalanced_time)

    # 整合不同offset的选股周期
    select = merge_period_offset(select, period_offset_df)

    # 就算持有周期内收益
    select = get_buy_sell_ret(all_factors_kline, select, rebalanced_time, c_rate, t_rate)

    # 标准化交易信息
    select_trade_info = get_trade_info(select, rebalanced_time)

    # 生成分析汇总表
    all_res, describe = analyze_stock_selection(select)
    describe.to_csv(save_path + "02_分析汇总.csv", encoding="gbk", header=False)

    ## 开始遍历每一行数据画图
    print("开始绘制个股行情图...")
    for i in all_res.index:
        # 获取币种名称
        stock_code = all_res.loc[i, "股票代码"]
        stock_name = all_res.loc[i, "股票名称"]
        print(f"正在绘制：第{i + 1}/{all_res.shape[0]}个 {stock_code}_{stock_name}")
        # 读取股票信息
        df = all_factors_kline[all_factors_kline["股票代码"] == stock_code]
        if "指数" in all_add_factor:
            df = pd.merge(left=df, right=index_data, on="交易日期", how="left", sort=True, indicator=True)

        # 截取时间
        df = df[(df["交易日期"] >= d_start) & (df["交易日期"] <= d_end)]
        # 获取所有的买入时间点
        open_times = [pd.to_datetime(time_range.split("--")[0]) for time_range in all_res.loc[i, "持有周期"]]
        # 获取所有的卖出时间点
        close_times = [pd.to_datetime(time_range.split("--")[1]) for time_range in all_res.loc[i, "持有周期"]]

        # 在数据中加入买入信息
        df.loc[df["交易日期"].isin(open_times), "买入时间"] = "买入"
        # 在数据中加入卖出信息
        df.loc[df["交易日期"].isin(close_times), "卖出时间"] = "卖出"

        # 产生交易表
        trade_df = select_trade_info[select_trade_info["股票代码"] == stock_code]
        _add_factor_main_list, _add_factor_sub_list = check_factor_in_df(
            df, _add_factor_main_list, _add_factor_sub_list
        )
        # 绘制中性策略的买卖信息
        pf.draw_hedge_signal_plotly(
            df,
            index_data,
            fig_save_path,
            f"{stock_code}_{stock_name}",
            trade_df,
            all_res.loc[i],
            _add_factor_main_list,
            _add_factor_sub_list,
            _color_dict,
        )

        file_path = os.path.join(fig_save_path, f"{stock_code}_{stock_name}.html")
        all_res.loc[i, "股票名称"] = f'=HYPERLINK("{file_path}","{stock_name}")'

    # 保存结果
    all_res.to_excel(save_path + "01_选股分析结果.xlsx", index=False)


if __name__ == "__main__":
    # ===== 策略信息配置 =====
    config = {
        # 回测结果名称，与config中一致
        "backtest_name": "选股测试",
        # 策略名称，输入格式：策略名称/选股结果 (可在回测结果文件夹下查看)
        # 这里的输入形式包括三种：1. config中的strategy_list的策略的位置信息，数字 0、1等 (每次只能输入一个数字)
        #                     2. config中的strategy_list的策略的规范名字，结构为 '#0.策略1'(#.{策略位置信息，数字表示}.{策略名字，对应name})
        #                     3. config中的strategy_list的策略的name，注意：如果strategy_list中策略的名字一样，代码默认读取第一个，例如name都为小市值，策略查看器代码默认读取第一个。
        "strategy_name": 0,  # 注意点：如果策略包含多个子策略，单次仅支持单个选币结果分析。
        "start_time": "2021-04-01",  # 分析开始时间
        "end_time": "2025-05-20",  # 分析结束时间
        # 主图增加(和股票K线图同一画布),均为折线图。
        "add_factor_main_list": [
            {"因子名称": "指数", "次坐标轴": True},
            {"因子名称": "factor_归母净利润同比增速_60", "次坐标轴": False},
            {"因子名称": "factor_市值", "次坐标轴": False},
        ],
        # 附图增加(在K线图下方展示)，一个dict为一个子图，因子名称的list大于1个值，则会被画在同一个图中，没用次坐标轴概念
        # 图形样式有且仅有三种选择K线图\柱状图\折线图
        "add_factor_sub_list": [
            # {'因子名称': ['换手率_20'], '图形样式': '折线图'},
            # {"因子名称": ["factor_归母净利润同比增速_60"], "图形样式": "折线图"},
            # {"因子名称": ["factor_市值"], "图形样式": "柱状图"},
            {"因子名称": ["factor_ROE_单季"], "图形样式": "折线图"}
        ],
        # ===== 以下信息几乎不需要配置 =====
        "add_days": 120,  # K线图需要提前/延长的天数，add_days指开始时间提前120天，结束时间往后延长120天
        # 按因子名称指定颜色，K线展示的内容固定颜色指定无效。
        # 颜色仅为plotly支持的颜色格式，基本上你知道的颜色相关的英文单词都有，没有会报错。 不指定颜色会随机配色
        "color_dict": {"指数": "red"},
    }

    # 运行主函数
    main(config)
