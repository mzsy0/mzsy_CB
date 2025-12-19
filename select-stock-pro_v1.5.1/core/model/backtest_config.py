"""
邢不行｜策略分享会
股票量化策略框架𝓟𝓻𝓸

版权所有 ©️ 邢不行
微信: xbx1717

本代码仅供个人学习使用，未经授权不得复制、修改或用于商业用途。

Author: 邢不行
"""

import os
import sys
from collections import defaultdict
from datetime import datetime
from itertools import product
from pathlib import Path
from types import ModuleType
from typing import Optional, List, Union, Dict

import pandas as pd

from config import runtime_data_path
from core.market_essentials import import_index_data, check_period_offset, download_period_offset
from core.model.rebalance_mode import RebalanceMode
from core.model.strategy_config import StrategyConfig, FactorConfig
from core.utils.factor_hub import FactorHub
from core.utils.log_kit import logger
from core.utils.path_kit import get_folder_path
from core.utils.strategy_hub import get_strategy_by_name


class BacktestConfig:
    def __init__(self, **config_dict: dict):
        # 账户名称，建议用英文，不要带有特殊符号
        self.name: str = config_dict.get("backtest_name", "默认策略回测")
        # 回测开始时间
        self.start_date: Optional[str] = config_dict.get("start_date", None)
        # 日期，为None时，代表使用到最新的数据，也可以指定日期，例如'2022-11-01'，但是指定日期
        self.end_date: Optional[str] = config_dict.get("end_date", None)

        # 策略列表，包含每个策略的详细配置
        self.strategy_list: List[StrategyConfig] = []
        self.strategy_name_list: List[str] = []
        self.strategy_list_raw: List[dict] = []
        # 初始资金默认100万
        self.initial_cash: float = config_dict.get("initial_cash", 100_0000)
        # 手续费，默认为0.002，表示万分之二
        self.c_rate: float = config_dict.get("c_rate", 1.2 / 10000)
        self.t_rate: float = config_dict.get("t_rate", 1 / 1000)  # 印花税，默认为0.001

        # 根据输入，进行一下重要中间变量的处理
        self.data_center_path: Path = Path(str(config_dict["data_center_path"]))
        # Rebalance 模式
        self.rebalance_mode: RebalanceMode = RebalanceMode.init(config_dict.get("rebalance_mode", None))
        # 整体资金使用率
        self.total_cap_usage: float = config_dict.get("total_cap_usage", 1)
        self.result_folder_name: str = config_dict.get("result_folder_name", "回测结果")

        # 如果你要diy的话，在这里设置你的数据中心路径
        # 股票日线数据，全量数据下载链接：https://www.quantclass.cn/data/stock/stock-trading-data
        self.stock_data_path: Path = self.data_center_path / "stock-trading-data-pro"
        # 指数数据路径，全量数据下载链接：https://www.quantclass.cn/data/stock/stock-main-index-data
        self.index_data_path: Path = self.data_center_path / "stock-main-index-data"
        # 其他的数据，全量数据下载链接：https://www.quantclass.cn/data/stock/stock-fin-data-xbx
        self.fin_data_path: Path = self.data_center_path / "stock-fin-data-xbx"

        self.has_fin_data: bool = self.fin_data_path.exists()  # 是否使用财务数据

        self.factor_params_dict: dict = {}  # 缓存因子参数，用于后续的因子聚合
        # 缓存分钟级因子参数，用于后续的因子聚合。2025-03-20添加分钟数据的支持 2025-04-17 修改格式
        self.factor_minutes_dict: dict[str, set[tuple]] = {}
        self.factor_col_name_list: List[str] = []
        self.hold_period_name_list: List[str] = []  # 持仓周期列表
        # 缓存分钟级数据的列表，包含换仓的分钟以及因子中包含的分钟节点
        self.min_data_list = []

        self.fin_cols: list = []  # 缓存财务因子列
        self.ov_cols: list = []  # 缓存全息数据的额外字段
        self.extra_data: dict = {}  # 缓存额外数据
        # 缓存被排除的板块
        self.excluded_boards: list = config_dict.get("excluded_boards", [])
        self.rebalance_time_list = []  # 需要用到分钟级rebalance_time时间的列表
        # 需要加载的分钟数据的级别，5分钟或者15分钟，默认为15分钟
        self.min_data_level = "1d"

        self.info = {}  # 用于缓存数据状态
        self.report: pd.DataFrame = pd.DataFrame()  # 回测报告

        # 遍历标记，用于遍历参数的时候，标记当前是第几个遍历
        # 遍历的INDEX，0表示非遍历场景，从1、2、3、4、...开始表示是第几个循环，当然也可以赋值为具体名称
        self.iter_round: Union[int, str] = 0
        # 遍历场景下，需要在原来文件路径基础上套一层回测名，在寻找最优参数.py中设置
        self.factory_backtest_name = "参数优化策略"

        self.period_offset_path = self.data_center_path / "period_offset.csv"

        if self.period_offset_path.exists():
            check_period_offset(self.period_offset_path)
        else:
            download_period_offset(self.period_offset_path)

        if all((self.stock_data_path.exists(), self.fin_data_path.exists(), self.index_data_path.exists())):
            pass  # 数据检查通过
        else:
            logger.critical(
                f"""必要数据有缺失，请检查:
1. {"🟢" if self.stock_data_path.exists() else "🔴"} {self.stock_data_path}
2. {"🟢" if self.fin_data_path.exists() else "🔴"} {self.fin_data_path}
3. {"🟢" if self.index_data_path.exists() else "🔴"} {self.index_data_path}
3. {"🟢" if self.period_offset_path.exists() else "🔴"} {self.period_offset_path}"""
            )
            sys.exit()

    @property
    def factor_minutes_list(self) -> set[str]:
        """
        所有因子的分钟数据（不包含调仓分钟数据）
        :return: set[str]
        示例：{"0945", "0955", "1015"}
        """
        result = set()
        for value in self.factor_minutes_dict.values():
            for item in value:
                if isinstance(item, tuple):
                    result.update(item)
                else:
                    result.add(item)
        # 如果数据量太大可以改成生成式加快效率（目前只有5分钟数据，全部用上也就48个，即便算上1分钟，也才240+48，数据量太小了）
        # result = set(
        #     sub_item
        #     for value in self.factor_minutes_dict.values()
        #     for item in value
        #     for sub_item in (item if isinstance(item, tuple) else [item])
        # )
        return result

    def desc(self):
        info_list = [
            "=" * 82,
            f"""🔵 {self.name}
→ 回测周期：{self.start_date} -> {self.end_date}
→ 初始资金：￥{self.initial_cash:,.2f}
→ 费率设置：手续费{self.c_rate * 10000:,.1f}‱, 印花税{self.t_rate * 1000:,.1f}‰
→ 数据设置:
  - 财务数据: {self.fin_cols if self.fin_cols else '∅ 否'}
  - 全息数据: {self.ov_cols if self.ov_cols else '∅ 否'}
  - 分钟数据: {self.min_data_list if self.min_data_list else '∅ 否'}，换仓时间：{self.rebalance_time_list if self.rebalance_time_list else '∅ 否'}
  - 外部数据: {list(self.extra_data.keys()) if self.extra_data else '∅ 否'}
→ 数据中心路径："{self.data_center_path}"
→ 结果路径："{self.get_result_folder()}"
→ 板块过滤：{self.excluded_boards}
→ 包含子策略：{'、'.join(self.strategy_name_list)}
→ 额外信息：{self.info}""",
        ]

        for strategy in self.strategy_list:
            info_list.append(f"  {strategy}")

        info_list.append("=" * 82 + "\n")

        return "\n".join(info_list)

    def save(self):
        pd.to_pickle(self, self.get_result_folder() / "config.pkl")

    # noinspection PyUnusedLocal
    def load_strategies(self, strategy_list: Union[list, tuple], timing_config=None):
        self.strategy_list_raw = strategy_list
        # 所有策略中的权重，当且仅当超过1的时候，才会做归一化处理
        all_cap_weight = max(sum(item.get("cap_weight", 1) for item in strategy_list), 1)
        merged_dict = defaultdict(list)  # 合并额外数据引用

        for index, stg_dict in enumerate(strategy_list):
            strategy_name = stg_dict["name"]
            strategy_info = stg_dict.pop("info", {})
            stg_dict["funcs"] = get_strategy_by_name(strategy_name)
            stg_dict["runtime_folder"] = self.get_runtime_folder()  # 运行过程中的文件夹
            stg_dict["result_folder"] = self.get_result_folder()  # 选股结果文件夹
            strategy = StrategyConfig.init(index, **stg_dict)
            if strategy.cap_weight < 1e-9:
                continue
            strategy.cap_weight = strategy.cap_weight / all_cap_weight  # 加权平均策略权重

            # 缓存持仓周期的事情
            self.hold_period_name_list += strategy.hold_period_name_list
            # 判断是否有额外的调仓时间
            self.rebalance_time_list += [
                reb_time for reb_time in strategy.rebalance_time.split("-") if reb_time not in ["open", "close"]
            ]

            self.strategy_list.append(strategy)
            self.strategy_name_list.append(strategy_name)
            self.factor_col_name_list += strategy.factor_columns

            self.info = strategy_info  # 缓存策略的状态信息，主要是应对单策略配置的模式

            # 针对当前策略的因子信息，整理之后的列名信息，并且缓存到全局
            for _factor in strategy.all_factors:
                # 添加到并行计算的缓存中
                self.factor_params_dict.setdefault(_factor.name, set()).add(_factor.param)
                # 2025-03-20添加分钟数据的支持 # 2025-04-17 修改格式
                self.factor_minutes_dict.setdefault(_factor.name, (set())).add(_factor.minutes)

                factor_ins = FactorHub.get_by_name(_factor.name)

                # 1. 合并财务因子
                self.fin_cols += getattr(factor_ins, "fin_cols", [])
                # 2. 合并全息数据的额外字段
                self.ov_cols += getattr(factor_ins, "ov_cols", [])
                # 3. 合并额外数据
                for k, v in getattr(factor_ins, "extra_data", {}).items():
                    merged_dict[k].extend(v)

        if len(self.strategy_list) == 0:
            logger.critical(f"没有读取到包含权重的策略，请检查策略配置")
            sys.exit(1)

        # 对列名进行去重
        self.fin_cols = list(sorted(set(self.fin_cols)))
        self.ov_cols = list(sorted(set(self.ov_cols)))
        self.extra_data = {key: list(set(value)) for key, value in sorted(merged_dict.items())}
        self.hold_period_name_list = list(sorted(set(self.hold_period_name_list)))
        self.factor_col_name_list = list(sorted(set(self.factor_col_name_list)))
        self.min_data_list = list(sorted(self.factor_minutes_list.union(self.rebalance_time_list)))
        self.rebalance_time_list = list(sorted(set(self.rebalance_time_list)))

        # 判断要用到什么级别的分钟数据
        if self.min_data_list:
            is_all_15min = all(minute[-2:] in ["45", "00", "15", "30"] for minute in self.min_data_list)
            self.min_data_level = "15m" if is_all_15min else "5m"

            self.extra_data[f"{self.min_data_level}in_close"] = list(
                set(self.min_data_list + self.extra_data.get(f"{self.min_data_level}in_close", []))
            )

            # 使用到分钟数据，回测时间需要从2010-01-01开始
            if self.start_date < "2010-01-01":
                logger.warning(
                    f"回测使用到分钟数据，应当从2010年开始，已经自动将回测起始时间从：{self.start_date}修改为2010-01-01"
                )
                self.start_date = "2010-01-01"

            # if timing_config:
        #     self.timing = TimingSignal(**timing_config)
        # 缓存交易日偏移，按照策略自动裁切

    def load_period_offset(self, auto_cols=True) -> pd.DataFrame:
        if self.hold_period_name_list and auto_cols:
            return pd.read_csv(
                self.period_offset_path,
                encoding="gbk",
                parse_dates=["交易日期"],
                skiprows=1,
                usecols=["交易日期"] + self.hold_period_name_list,
            )
        else:
            return pd.read_csv(self.period_offset_path, encoding="gbk", parse_dates=["交易日期"], skiprows=1)

    def load_index_data(self, use_range=False):
        """
        加载指数数据
        index_data (DataFrame): 合并后的指数数据
        """
        if use_range:
            return import_index_data(self.index_data_path / "sh000001.csv", [self.start_date, self.end_date])
        else:
            # 2025-03-25 10:48:09和夏普确认，我们回测研究时候，历史指数数据从2007年开始
            return import_index_data(self.index_data_path / "sh000001.csv", ["2007-01-01", None])

    def read_trading_dates(self, first_date, last_date):
        period_offset = self.load_period_offset()
        trading_dates = period_offset["交易日期"]

        # 支持一下开、闭区间的设定
        if first_date:
            trading_dates = trading_dates[trading_dates >= first_date]
        if last_date:
            trading_dates = trading_dates[trading_dates <= last_date]
        # trading_dates = trading_dates[(trading_dates >= first_date) & (trading_dates <= last_date)]
        return trading_dates

    def get_result_folder(self) -> Path:
        if self.iter_round == 0:
            return get_folder_path(runtime_data_path, self.result_folder_name, self.name)
        else:
            config_name = f"策略组_{self.iter_round}" if isinstance(self.iter_round, int) else self.iter_round
            if self.name.startswith(f"S{self.iter_round}"):
                config_name = self.name
            return get_folder_path(runtime_data_path, "遍历结果", self.factory_backtest_name, config_name)

    def get_cache_folder(self, folder_name):
        if self.iter_round == 0:
            return get_folder_path(runtime_data_path, folder_name, self.name)
        else:
            return get_folder_path(runtime_data_path, folder_name, self.factory_backtest_name)

    def get_runtime_folder(self):
        return self.get_cache_folder("运行缓存")

    @staticmethod
    def get_analysis_folder() -> Path:
        return get_folder_path(runtime_data_path, "分析结果")

    def get_fullname(self, as_folder_name=False):
        fullname_list = [self.name]
        for stg in self.strategy_list:
            fullname_list.append(str(stg))

        fullname = " ".join(fullname_list) + f"，初始资金￥{self.initial_cash * self.total_cap_usage:,.2f}"
        return f"{self.name}" if as_folder_name else fullname

    def set_report(self, report: pd.DataFrame):
        report["param"] = self.get_fullname()
        self.report = report

    def get_strategy_config_sheet(self, with_factors=True, sep_filter=False) -> dict:
        factor_dict = {"持仓周期": [], "选股数量": []}
        for stg in self.strategy_list:
            factor_dict["持仓周期"].append(stg.hold_period_name_list)
            factor_dict["选股数量"].append(stg.select_num)

            for factor_config in stg.all_factors:
                if sep_filter:
                    factor_type = "因子" if isinstance(factor_config, FactorConfig) else "过滤"
                    _name = f"#{factor_type}-{factor_config.name}"
                else:
                    _name = f"#因子-{factor_config.name}"
                _val = factor_config.param
                if _name not in factor_dict:
                    factor_dict[_name] = []
                factor_dict[_name].append(_val)
        ret = {"策略": self.name, "策略详情": self.get_fullname()}
        if with_factors:
            ret.update(**{k: "，".join(map(str, v)) for k, v in factor_dict.items()})

        # if self.timing:
        #     ret['再择时'] = str(self.timing)
        return ret

    def get_final_equity_path(self):
        # has_timing_signal = isinstance(self.timing, TimingSignal)
        # if has_timing_signal:
        #     filename = '资金曲线_再择时.csv'
        # else:
        filename = "资金曲线.csv"
        final_equity_path = self.get_result_folder() / filename
        return final_equity_path

    def get_period_weights(self) -> Dict[str, float]:
        weight = {hold_period: 0 for hold_period in self.hold_period_name_list}
        for strategy in self.strategy_list:
            for hold_period in strategy.hold_period_name_list:
                weight[hold_period] += strategy.cap_weight / len(strategy.offset_list)
        return weight

    @classmethod
    def init_from_config(cls, backtest_name=None, load_strategy_list=True, real_trading=False) -> "BacktestConfig":
        import config

        # 提取自定义变量
        config_dict = {
            key: value
            for key, value in vars(config).items()
            if not key.startswith("__") and not isinstance(value, ModuleType)
        }
        if backtest_name:
            config_dict["backtest_name"] = backtest_name
        conf = cls(**config_dict)

        if not real_trading:
            # Rebalance 模式，实盘中禁用
            conf.rebalance_mode = RebalanceMode.init(config_dict.get("rebalance_mode", None))
        if load_strategy_list:
            # 是否自动加载策略，默认会初始化策略列表
            conf.load_strategies(config.strategy_list, getattr(config, "re_timing", None))
        return conf

    @classmethod
    def init_with_stg_config(cls, stg_config: dict, backtest_name=None, factory_info=None) -> "BacktestConfig":
        """
        通过输入的config初始化，并且自动加载对应的策略列表
        :param stg_config:
        :param backtest_name:
        :param factory_info:
        :return:
        """
        # 针对单策略模式做充分兼容
        if "strategy_list" not in stg_config:
            stg_config = dict(name=stg_config["name"], strategy_list=[stg_config])
        backtest_config = BacktestConfig.init_from_config(backtest_name, load_strategy_list=False, real_trading=False)

        # 如果factory_info不为None，则设置为iter_round。需要前置判断，避免load_strategies中文件夹路径错误
        if factory_info:
            backtest_config.iter_round = factory_info["iter_round"]
            backtest_config.factory_backtest_name = factory_info["backtest_name"]

        # 加载策略
        backtest_config.load_strategies(stg_config["strategy_list"], stg_config.get("re_timing", None))

        return backtest_config

    @property
    def select_results_path(self):
        filename = "选股结果.pkl"
        return self.get_result_folder() / filename


class BacktestConfigFactory:
    """
    遍历参数的时候，动态生成配置
    """

    def __init__(self, **conf):
        # ====================================================================================================
        # ** 参数遍历配置 **
        # 可以指定因子遍历的参数范围
        # ====================================================================================================
        # 存储生成好的config list和strategy list
        self.config_list: List[BacktestConfig] = []
        self.backtest_name = conf.get("backtest_name")

        if not self.backtest_name:
            self.backtest_name = f'默认策略-{datetime.now().strftime("%Y%m%dT%H%M%S")}'

        # 缓存全局配置
        self.is_use_spot = conf.get("is_use_spot", False)
        self.black_list = conf.get("black_list", set())

        # 存储生成好的config list和strategy list
        self.strategy_list: List[StrategyConfig] = []

    @property
    def result_folder(self) -> Path:
        return get_folder_path(runtime_data_path, "遍历结果", self.backtest_name)

    def generate_all_factor_config(self):
        """
        产生一个conf，拥有所有策略的因子，用于因子加速并行计算
        """
        backtest_config = BacktestConfig.init_from_config(
            self.backtest_name, load_strategy_list=False, real_trading=False
        )
        strategy_list = []
        for conf in self.config_list:
            strategy_list.extend(conf.strategy_list_raw)
        backtest_config.load_strategies(strategy_list)

        return backtest_config

    def get_name_params_sheet(self) -> pd.DataFrame:
        rows = []
        for config in self.config_list:
            rows.append(config.get_strategy_config_sheet())

        sheet = pd.DataFrame(rows)
        sheet.to_excel(self.config_list[-1].get_result_folder().parent / "策略回测参数总表.xlsx", index=False)
        return sheet

    def generate_configs_by_strategies(self, strategies: List[list], timing_strategies=None) -> List[BacktestConfig]:
        config_list = []
        iter_round = 0

        if not timing_strategies:
            timing_strategies = [None]

        self.backtest_name = self.backtest_name or "默认参数遍历"

        for strategy_list, timing_config in product(strategies, timing_strategies):
            iter_round += 1

            backtest_name = f"S{iter_round}-{self.backtest_name}"
            backtest_config = BacktestConfig.init_with_stg_config(
                # 传入的strategy_list是list of list，需要转换为list of dict，选股策略框架中，名称就是配置中的backtest_name
                {"strategy_list": strategy_list, "re_timing": timing_config, "name": self.backtest_name},
                backtest_name,
                factory_info={"iter_round": iter_round, "backtest_name": self.backtest_name},
            )

            config_list.append(backtest_config)

        self.config_list = config_list

        return config_list


def load_config(real_trading=False) -> BacktestConfig:
    if os.getenv("FUEL_CLIENT_CONFIG_PATH"):
        real_trading = True
    return BacktestConfig.init_from_config(real_trading=real_trading)


def create_factory(strategies, backtest_name=None):
    if backtest_name is None:
        from config import backtest_name
    factory = BacktestConfigFactory(backtest_name=backtest_name)
    factory.generate_configs_by_strategies(strategies)

    return factory
