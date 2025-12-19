"""
邢不行｜策略分享会
股票量化策略框架𝓟𝓻𝓸

版权所有 ©️ 邢不行
微信: xbx1717

本代码仅供个人学习使用，未经授权不得复制、修改或用于商业用途。

Author: 邢不行
"""

import re
from dataclasses import dataclass
from functools import cached_property
from typing import List, Union


def get_col_name(factor_name, factor_param, minutes=()):
    col_name = f"{factor_name}"
    if factor_param:  # 如果参数有意义的话才显示出来
        if isinstance(factor_param, (tuple, list)):
            factor_param_str = "(" + ",".join(map(str, factor_param)) + ")"
        else:
            factor_param_str = str(factor_param)
        col_name += f"_{factor_param_str}"
    if minutes:  # 只有配置了分钟因子的数据，才显示出来
        col_name += "_" + ",".join(map(str, minutes))
    return col_name


# 自定义一个类来保持dict的使用方法，并保证其可哈希，且保证顺序
class HashableDict:
    def __init__(self, data: dict):
        # 将字典按键排序并转为tuple，保证顺序并可哈希
        self.data = tuple(sorted(data.items()))

    def __repr__(self):
        # 使其返回一个类似字典的表示方式
        if isinstance(self.data, tuple):
            return "(" + ",".join(f"{k}={v}" for k, v in self.data) + ")"
        return repr(self.data)

    def __eq__(self, other):
        return self.data == other.data

    def __hash__(self):
        return hash(self.data)

    # 支持通过 [] 方式访问
    def __getitem__(self, key):
        if isinstance(self.data, tuple):
            # 将tuple转换回一个dict来支持按键访问
            dict_data = dict(self.data)
            return dict_data[key]
        else:
            raise TypeError(f"Cannot subscript a {type(self.data)} object")


def parse_param(param) -> Union[tuple, HashableDict, str, int, float, bool, None]:
    # param的类型需要转换为hashable的状态
    if isinstance(param, list):
        param = tuple(param)
    elif isinstance(param, dict):
        param = HashableDict(param)
    elif isinstance(param, (str, int, float, tuple, bool)) or param is None:
        pass
    else:
        raise ValueError(f"不支持的参数类型：{type(param)}")
    return param


@dataclass(frozen=True)
class FactorConfig:
    name: str = "Factor"  # 选股因子名称
    is_sort_asc: bool = True  # 是否正排序
    param: Union[tuple, HashableDict, str, int, float, bool, None] = 3  # 选股因子参数
    args: Union[tuple, HashableDict, str, int, float, bool, None] = 1  # 默认是选股因子权重，也可以是计算因子时候的参数
    minutes: tuple = ()  # 选股因子的分钟级别，2025-03-20之前的版本，只支持4个因子，需要补全分钟级别的参数

    @classmethod
    def parse_list(cls, factor_list: list, not_weight=False) -> List:
        all_long_factor_weight = (
            0 if not_weight else max(sum([factor[3] for factor in factor_list]), 1)
        )  # 小于1的时候不做归一化

        parsed_factor_list = []
        for factor_tuple in factor_list:
            # 2025-03-20之前的版本，只支持4个因子，需要补全分钟级别的参数
            if len(factor_tuple) == 4:
                factor_name, is_sort_asc, param, args = factor_tuple
                minutes = ()  # 默认是没有分钟级别的
            else:
                factor_name, is_sort_asc, param, args, minutes = factor_tuple
                if isinstance(minutes, str):
                    minutes = (minutes,)
                else:
                    minutes = tuple(minutes)

            # param的类型需要转换为hashable的状态
            p_param = parse_param(param)

            # args的相关处理，默认是作为因子权重，但是股票场景下，是可以拓展的
            if not_weight:
                p_args = parse_param(args)
            else:
                p_args = args / all_long_factor_weight

            _factor = cls(name=factor_name, is_sort_asc=is_sort_asc, param=p_param, args=p_args, minutes=minutes)
            parsed_factor_list.append(_factor)
        return parsed_factor_list

    @cached_property
    def is_min_factor(self):
        return len(self.minutes) > 0

    @cached_property
    def col_name(self):
        return get_col_name(self.name, self.param, self.minutes)

    @property
    def weight(self):
        return float(self.args)  # 当使用自定义函数的时候，可以通过这个别名变量，来获取对应的数值

    def __repr__(self):
        return f'{self.col_name}{"↑" if self.is_sort_asc else "↓"}#{self.args}'

    def to_tuple(self):
        return self.name, self.is_sort_asc, self.param, self.args


@dataclass(frozen=True)
class FilterMethod:
    how: str = ""  # 过滤方式
    range: str = ""  # 过滤值

    def __repr__(self):
        match self.how:
            case "rank":
                name = "排名"
            case "pct":
                name = "排名百分比"
            case "val":
                name = "数值"
            case _:
                raise ValueError(f"不支持的过滤方式：`{self.how}`")

        return f"{name}:{self.range}"

    def to_val(self):
        return f"{self.how}:{self.range}"


@dataclass(frozen=True)
class FilterFactorConfig:
    name: str = "Bias"  # 选股因子名称
    param: Union[tuple, HashableDict, str, int, float, bool, None] = 3  # 选股因子参数
    method: FilterMethod = None  # 过滤方式
    is_sort_asc: bool = True  # 是否正排序
    minutes: tuple = ()  # 选股因子的分钟级别，2025-03-20之前的版本，只支持4个因子，需要补全分钟级别的参数

    def __repr__(self):
        _repr = self.col_name
        if self.method:
            _repr += f'{"↑" if self.is_sort_asc else "↓"}#{self.method}'
        return _repr

    @cached_property
    def col_name(self):
        return get_col_name(self.name, self.param, self.minutes)

    @classmethod
    def init(cls, filter_factor: tuple):
        # 仔细看，结合class的默认值，这个和默认策略中使用的过滤是一模一样的
        config = dict(name=filter_factor[0], param=parse_param(filter_factor[1]))
        if len(filter_factor) > 2:
            # 可以自定义过滤方式
            _how, _range = re.sub(r"\s+", "", filter_factor[2]).split(":")
            config["method"] = FilterMethod(how=_how, range=_range)
        if len(filter_factor) > 3:
            # 可以自定义排序
            config["is_sort_asc"] = filter_factor[3]
        return cls(**config)

    def to_tuple(self, full_mode=False):
        if full_mode:
            return self.name, self.param, self.method.to_val(), self.is_sort_asc
        else:
            return self.name, self.param
