"""
邢不行｜策略分享会
股票量化策略框架𝓟𝓻𝓸

版权所有 ©️ 邢不行
微信: xbx1717

本代码仅供个人学习使用，未经授权不得复制、修改或用于商业用途。

Author: 邢不行
"""

from dataclasses import dataclass, field
from typing import List, Dict, Callable, Union

from core.model.factor_config import FactorConfig, parse_param


@dataclass
class TimingSignal:
    # 信号名称
    name: str = "TimingSignal"
    # 因子计算的股票范围 例如 100 表示复合因子前50个股票择时，0.5 表示前50%的股票择时，0 表示全部股票择时（不建议）；
    limit: Union[int, float] = 100
    # 信号因子
    factor_list: List[FactorConfig] = field(default_factory=list)
    # 信号参数
    params: Union[tuple, float, int, str] = ()
    # 信号时间
    signal_time: str = "close"
    # 回溯多久的历史数据，因子rolling越大，参数越大，速度也会越慢
    recall_days: int = 256

    # **fallback仓位**，当定风波信号因为各种原因在换仓前无法执行的时候，比如计算超时，会使用这个仓位。
    # 1 表示到了换仓时间，没有算出来就全部出击。也可以是 0 表示不出击。也可以选 0.5 表示出击一半仓位。
    # 默认是 -1 表示按照因子计算和择时逻辑走，不使用 fallback_position（目前夏普提供的策略大部分是出击）
    fallback_position: Union[int, float] = -1

    # 策略函数
    funcs: Dict[str, Callable] = field(default_factory=dict)

    @classmethod
    def init(cls, **config):
        config["factor_list"] = FactorConfig.parse_list(config.get("factor_list", []), False)
        config["params"] = parse_param(config.get("params", ()))
        timing_signal = cls(**config)

        if timing_signal.min_list:  # 有分钟数据的因子的话，会自动获取最大值，否则默认为close
            timing_signal.signal_time = max(timing_signal.min_list)

        return timing_signal

    @property
    def min_list(self):
        _min_list = [m for f in self.factor_list for m in f.minutes if str(m).isdigit()]
        return tuple(sorted(set(_min_list)))

    def __repr__(self) -> str:
        _str = f"{self.name}_{self.signal_time}，择时范围{self.limit}，因子{self.factor_list}，参数{self.params}"
        if self.fallback_position >= 0:
            _str += f"，fallback仓位{self.fallback_position}"
        return _str
