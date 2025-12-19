"""
邢不行｜策略分享会
股票量化策略框架𝓟𝓻𝓸

版权所有 ©️ 邢不行
微信: xbx1717

本代码仅供个人学习使用，未经授权不得复制、修改或用于商业用途。

Author: 邢不行
"""

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import List, Dict, Callable, Tuple, Optional, Union

import numpy as np
import pandas as pd

from config import days_listed, runtime_data_path
from core.model.factor_config import FilterFactorConfig, FactorConfig
from core.model.timing_signal import TimingSignal
from core.utils.signal_hub import get_signal_by_name

ALLOWED_OFFSETS = [
    "2_0",
    "2_1",
    "3_0",
    "3_1",
    "3_2",
    "4_0",
    "4_1",
    "4_2",
    "4_3",
    "5_0",
    "5_1",
    "5_2",
    "5_3",
    "5_4",
    "10_0",
    "10_1",
    "10_2",
    "10_3",
    "10_4",
    "10_5",
    "10_6",
    "10_7",
    "10_8",
    "10_9",
    "W_0",
    "W_1",
    "W_2",
    "W_3",
    "W_4",
    "2W_0",
    "2W_1",
    "3W_0",
    "3W_1",
    "3W_2",
    "4W_0",
    "4W_1",
    "4W_2",
    "4W_3",
    "5W_0",
    "5W_1",
    "5W_2",
    "5W_3",
    "5W_4",
    "6W_0",
    "6W_1",
    "6W_2",
    "6W_3",
    "6W_4",
    "6W_5",
    "M_0",
    "M_-5",
    "W53_0",
]


def calc_factor_common(df, factor_list: List[FactorConfig]):
    factor_val = np.zeros(df.shape[0])
    for factor_config in factor_list:
        # 计算单个因子的排名
        _rank = df.groupby("交易日期")[factor_config.col_name].rank(ascending=factor_config.is_sort_asc, method="min")
        # 将因子按照权重累加
        factor_val += _rank * factor_config.weight
    return factor_val


def filter_series_by_range(series, range_str):
    # 提取运算符和数值
    operator = range_str[:2] if range_str[:2] in [">=", "<=", "==", "!="] else range_str[0]
    value = float(range_str[len(operator) :])

    match operator:
        case ">=":
            return series >= value
        case "<=":
            return series <= value
        case "==":
            return series == value
        case "!=":
            return series != value
        case ">":
            return series > value
        case "<":
            return series < value
        case _:
            raise ValueError(f"Unsupported operator: {operator}")


def filter_common(df, filter_list):
    condition = pd.Series(True, index=df.index)

    for filter_config in filter_list:
        col_name = filter_config.col_name
        match filter_config.method.how:
            case "rank":
                rank = df.groupby("交易日期")[col_name].rank(ascending=filter_config.is_sort_asc, pct=False)
                condition = condition & filter_series_by_range(rank, filter_config.method.range)
            case "pct":
                rank = df.groupby("交易日期")[col_name].rank(ascending=filter_config.is_sort_asc, pct=True)
                condition = condition & filter_series_by_range(rank, filter_config.method.range)
            case "val":
                condition = condition & filter_series_by_range(df[col_name], filter_config.method.range)
            case _:
                raise ValueError(f"不支持的过滤方式：{filter_config.method.how}")

    return condition


@dataclass
class StrategyConfig:
    name: str = "Strategy"

    # 持仓周期。
    hold_period: str = "W"

    # 持仓周期的参数，比如offset
    offset_list: Tuple[int] = (0,)

    # 策略权重
    cap_weight: float = 1.0

    # 原始数据的周期。
    candle_period: str = "D"

    # 选股数量。1 表示一个股票; 0.1 表示做多10%的股票
    select_num: Union[int, float] = 0.1

    # ** 换仓时间 **
    # 选股日换仓的时候，我们可以自定义换仓的时间点
    # - 'close-open'：选股日收盘前卖出，交易日开盘后买入（隔日换仓）；
    # - 'open'：交易日开盘后先卖出，交易日开盘后再买入（日内早盘）；
    # - 'close'：选股日收盘前卖出，选股日收盘前再买入（日内尾盘）；
    # 默认是 'close-open'，表示收盘买，下个开盘买，即隔日换仓
    rebalance_time: str = "close-open"

    # 选股过程中最终用于股票排名的因子名
    factor_name: str = "复合因子"

    # 因子名（和factors文件中相同），排序方式，参数，权重。
    factor_list: List[FactorConfig] = field(default_factory=list)

    filter_list: List[FilterFactorConfig] = field(default_factory=list)

    # 策略函数
    funcs: Dict[str, Callable] = field(default_factory=dict)

    # 择时信号
    timing: Optional[TimingSignal] = None

    # 运行过程中的文件夹，依赖于 backtest 在初始化时传入
    runtime_folder: Path = field(default_factory=Path)

    # 选股结果文件夹，依赖于 backtest 在初始化时传入
    result_folder: Path = field(default_factory=Path)

    @cached_property
    def period_type(self) -> str:
        return self.hold_period[-1]

    @cached_property
    def period_num(self) -> int:
        num_str = self.hold_period[:-1]

        if num_str.isnumeric():
            return int(num_str)
        else:
            return 1

    @cached_property
    def hold_period_name_list(self) -> List[str]:
        match self.period_type:
            case "D":
                period_prefix = f"{self.period_num}_"
            case "M":
                period_prefix = f"M_"
            case _:
                period_prefix = f"{self.hold_period}_"

        if period_prefix.startswith("1W_"):
            period_prefix = period_prefix.replace("1W_", "W_")
        return [f"{period_prefix}{offset}" for offset in self.offset_list]

    @cached_property
    def hold_period_name(self) -> str:
        return ",".join(self.hold_period_name_list)

    @cached_property
    def factor_columns(self) -> List[str]:
        factor_columns = set()  # 去重

        # 针对当前策略的因子信息，整理之后的列名信息，并且缓存到全局
        for factor_config in self.factor_list:
            # 策略因子最终在df中的列名
            factor_columns.add(factor_config.col_name)  # 添加到当前策略缓存信息中

        # 针对当前策略的过滤因子信息，整理之后的列名信息，并且缓存到全局
        for filter_factor in self.filter_list:
            # 策略过滤因子最终在df中的列名
            factor_columns.add(filter_factor.col_name)  # 添加到当前策略缓存信息中

        # 针对当前策略的过滤因子信息，整理之后的列名信息，并且缓存到全局
        for timing_factor in self.timing.factor_list if self.timing is not None else ():
            # 策略过滤因子最终在df中的列名
            factor_columns.add(timing_factor.col_name)

        return list(factor_columns)

    @cached_property
    def all_factors(self) -> set:
        all_factors = set()
        for factor_config in self.factor_list:
            all_factors.add(factor_config)
        for filter_factor in self.filter_list:
            all_factors.add(filter_factor)
        for timing_factor in self.timing.factor_list if self.timing else []:
            all_factors.add(timing_factor)
        return all_factors

    @classmethod
    def init(cls, index: int, **config):
        is_custom_select = "calc_select_factor" in config["funcs"]
        config["factor_list"] = FactorConfig.parse_list(config.get("factor_list", []), is_custom_select)
        config["filter_list"] = [
            FilterFactorConfig.init(filter_config) for filter_config in config.get("filter_list", [])
        ]
        timing_config = config.get("timing", {})
        if timing_config:
            timing_config["funcs"] = get_signal_by_name(timing_config["name"])
            config["timing"] = TimingSignal.init(**timing_config)

        stg_conf = cls(**config)
        stg_conf.name = f"#{index}.{stg_conf.name}"

        return stg_conf

    def __repr__(self):
        return f"{self.cap_weight * 100:.2f}%{self.name}，周期{self.hold_period_name}，{self.select_num}个，因子{self.factor_list}，过滤{self.filter_list}，{self.trade_mode_name()}㇑{self.timing if self.timing else '无择时'}"

    def trade_mode_name(self):
        match self.rebalance_time:
            case "close-open":
                return "隔日换仓"
            case "close":
                return "日内尾盘"
            case "open":
                return "日内早盘"
            case _:
                sell_time, buy_time = self.rebalance_time.split("-")
                return "自定义换仓({}卖{}买)".format(sell_time, buy_time)

    def max_int_param(self) -> int:
        max_int = 0
        for factor_config in self.all_factors:
            if isinstance(factor_config.param, int):
                max_int = max(max_int, factor_config.param)
        return max_int

    def filter_before_select(self, period_df):
        if "filter_stock" in self.funcs:
            return self.funcs["filter_stock"](period_df, self)

        # 通用的filter筛选
        # =删除不能交易的周期数
        # 删除月末为st状态的周期数
        cond1 = ~period_df["股票名称"].str.contains("ST", regex=False)
        # 删除月末为s状态的周期数
        cond2 = ~period_df["股票名称"].str.contains("S", regex=False)
        # 删除月末有退市风险的周期数
        cond3 = ~period_df["股票名称"].str.contains("*", regex=False)
        cond4 = ~period_df["股票名称"].str.contains("退", regex=False)
        # 删除交易天数过少的周期数
        # cond5 = period_df['交易天数'] / period_df['市场交易天数'] >= 0.8

        cond6 = period_df["下日_是否交易"] == 1
        cond7 = period_df["下日_开盘涨停"] != 1
        cond8 = period_df["下日_是否ST"] != 1
        cond9 = period_df["下日_是否退市"] != 1
        cond10 = period_df["上市至今交易天数"] > days_listed

        # common_filter = cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond7 & cond8 & cond9 & cond10
        common_filter = cond1 & cond2 & cond3 & cond4 & cond6 & cond7 & cond8 & cond9 & cond10
        period_df = period_df[common_filter]

        filter_condition = filter_common(period_df, self.filter_list)

        return period_df[filter_condition]

    def calc_select_factor(self, period_df):
        if "calc_select_factor" in self.funcs:
            return self.funcs["calc_select_factor"](period_df, self)
        period_df[self.factor_name] = self.calc_select_factor_default(period_df)
        return period_df

    def calc_select_factor_default(self, period_df):
        return calc_factor_common(period_df, self.factor_list)

    def calc_signal(self, factor_df: pd.DataFrame, mode="backtest") -> pd.DataFrame:
        """
        目前是：前置过滤后的界面DataFrame
        :param factor_df: 前置过滤后的截面DataFrame
        :param mode: 运行模式
        :return: 择时信号DataFrame
        """
        # ======================== 处理选股范围 ===========================
        if self.timing.limit > 0:
            # 是否是百分比
            pct = self.timing.limit < 1
            factor_rank = factor_df.groupby("交易日期")[self.factor_name].rank(method="min", ascending=True, pct=pct)
            # 选取排名靠前的股票
            df_after_limit = factor_df[factor_rank <= self.timing.limit]
        else:  # 全部股票，stock_range小于0时，表示全部股票
            df_after_limit = factor_df

        # 如果有缓存的话拼接一下历史数据
        hist_df_path = self.get_trade_info_path().parent / f"{self.name}_择时行情数据.pkl"
        if (mode != "backtest") and hist_df_path.exists():
            # 读入历史数据
            hist_df = pd.read_pickle(hist_df_path)
            # 取出需要拼接的列
            limited_cols = [col for col in hist_df.columns if col in df_after_limit.columns]

            # 拼接历史数据和最新数据，并且保持排序（会复制一份，避免污染）
            df_after_limit = pd.concat(
                [hist_df, df_after_limit[limited_cols].copy()], ignore_index=True, sort=True, copy=False
            )

            # 按照日期、股票代码排序，自动填充factor需要的非择时期间计算的数据
            df_after_limit.sort_values(["交易日期", "股票代码"], inplace=True)
            df_after_limit.ffill(inplace=True)

            df_after_limit.drop_duplicates(["交易日期", "股票代码"], keep="last", inplace=True)
        df_after_limit.to_pickle(hist_df_path)

        signals = self.timing.funcs["signal"](self, df_after_limit)

        # signals 最后一行用 fallback_position 填充，如果fallback_position小于0，则不填充
        if not signals.empty and self.timing.fallback_position >= 0:
            signals.iloc[-1, signals.columns.get_loc("择时信号")] = self.timing.fallback_position

        # 保存实盘需要的交易信息
        stock_list = df_after_limit[df_after_limit["交易日期"] == df_after_limit["交易日期"].max()][
            "股票代码"
        ].to_list()  # 选取最后一个交易日的股票代码
        # 选取最大的分钟数据
        time_str = max(self.timing.min_list) if self.timing.min_list else "close"
        self.save_trade_info("早盘择时", [time_str, stock_list])

        return signals

    def get_today_signal_path(self, root=runtime_data_path) -> Path:
        today_str = pd.Timestamp.today().strftime("%Y-%m-%d")
        if not isinstance(root, Path):
            root = Path(root)
        folder = root / "实盘信息" / today_str
        folder.mkdir(exist_ok=True, parents=True)
        return folder / f"{self.name}_信号.pkl"

    def save_today_signal(self, signal: pd.DataFrame):
        signal.to_pickle(self.get_today_signal_path())

    def get_trade_info_path(self):
        path = runtime_data_path / "实盘信息" / f"{self.name}.pkl"
        path.parent.mkdir(exist_ok=True, parents=True)  # 创建文件夹
        return path

    def save_trade_info(self, key, value):
        # 读取实盘信息
        save_path = self.get_trade_info_path()
        trade_info = self.read_trade_info()

        # 存储实盘信息
        trade_info[key] = value
        pd.to_pickle(trade_info, save_path)

    def read_trade_info(self, key=None):
        save_path = self.get_trade_info_path()
        trade_info = pd.read_pickle(save_path) if save_path.exists() else {}
        if key:
            return trade_info.get(key, None)
        else:
            return trade_info
