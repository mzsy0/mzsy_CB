"""
邢不行｜策略分享会
股票量化策略框架𝓟𝓻𝓸

版权所有 ©️ 邢不行
微信: xbx1717

本代码仅供个人学习使用，未经授权不得复制、修改或用于商业用途。

Author: 邢不行
"""

import itertools
import time
import warnings
import pandas as pd

from core.backtest import run_backtest_multi
from core.model.backtest_config import create_factory
from core.version import version_prompt

# ====================================================================================================
# ** 脚本运行前配置 **
# 主要是解决各种各样奇怪的问题们
# ====================================================================================================
warnings.filterwarnings("ignore")  # 过滤一下warnings，不要吓到老实人

# pandas相关的显示设置，基础课程都有介绍
pd.set_option("expand_frame_repr", False)  # 当列太多时不换行
pd.set_option("display.unicode.ambiguous_as_wide", True)  # 设置命令行输出时的列对齐功能
pd.set_option("display.unicode.east_asian_width", True)


def dict_itertools(dict_):
    keys = list(dict_.keys())
    values = list(dict_.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def __list_to_range(lst):
    """list列表逆推回range"""
    if len(lst) < 2:
        return lst  # 无法转换单元素列表

    step = lst[1] - lst[0]
    for i in range(1, len(lst)):
        if lst[i] - lst[i - 1] != step:
            return lst  # 步长不一致，无法转换

    start = lst[0]
    stop = lst[-1] + step
    return range(start, stop, step)


def __save_batch_params(params, path="参数.txt", filter_len=10):
    """按肉眼看到的样子将batch参数保存成txt，方便直接复制batch调优"""
    with open(path, "w") as f:
        f.write("batch = {\n")
        for k, v in params.items():
            if len(v) > filter_len:  # 当列表长度超过filter_len，才进行反推变成range
                v = __list_to_range(v)
            f.write(f"    '{k}': {v},\n")
        f.write("}\n")


if __name__ == "__main__":
    version_prompt()
    print(f"🌀 系统启动中，稍等...")
    r_time = time.time()
    # ====================================================================================================
    # 1. 配置需要遍历的参数
    # ====================================================================================================
    trav_name = "选股策略混合"
    batch = {
        "rebalance_time": ["0935-0945", "0945-0955"],
        "params": [0.5, 0.6],
        "limit": [200, 500],
        "开盘至今涨幅择时": ["0945"],
    }
    # 因子遍历的参数范围
    strategies = []
    for params_dict in dict_itertools(batch):
        strategy_list = [
            {
                "name": "小市值_基本面优化",
                "hold_period": "3D",
                "offset_list": [0, 1, 2],
                "select_num": 5,
                "cap_weight": 1,
                "rebalance_time": params_dict["rebalance_time"],
                "factor_list": [("市值", True, None, 1), ("归母净利润同比增速", False, 60, 1)],
                "filter_list": [("ROE", "单季", "pct:<=0.8", False)],
                "timing": {
                    "name": "定风波1P5择时",  # 择时策略名称
                    "limit": params_dict["limit"],
                    "factor_list": [
                        ("开盘至今涨幅", False, None, 1, params_dict["开盘至今涨幅择时"]),
                        ("隔夜涨跌幅", False, None, 1, "开盘价"),
                    ],
                    "params": params_dict["params"],
                },
            }
        ]
        strategies.append(strategy_list)

    # ====================================================================================================
    # 2. 生成策略配置
    # ====================================================================================================
    print(f"🌀 生成策略配置...")
    backtest_factory = create_factory(strategies, backtest_name=trav_name)

    # ====================================================================================================
    # 3. 寻找最优参数
    # ====================================================================================================
    # boost为True：并行选股；boost为False：串行选股
    # 第一次运行，且不太确定的时候，可以考虑使用 `boost=False`，回测组不多的时候，不会慢太多的哈~
    report_list = run_backtest_multi(backtest_factory, boost=True)

    # ====================================================================================================
    # 4. 根据回测参数列表，展示最优参数
    # ====================================================================================================
    s_time = time.time()
    print(f"🌀 展示最优参数...")
    all_params_map = pd.concat(report_list, ignore_index=True)
    report_columns = all_params_map.columns  # 缓存列名

    # 合并参数细节
    sheet = backtest_factory.get_name_params_sheet()
    all_params_map = all_params_map.merge(sheet, left_on="param", right_on="策略详情", how="left")

    # 按照累积净值排序，并整理结果
    all_params_map.sort_values(by="累积净值", ascending=False, inplace=True)
    all_params_map = all_params_map[[*sheet.columns, *report_columns]].drop(columns=["param"])
    all_params_map.to_excel(backtest_factory.result_folder / f"最优参数.xlsx", index=False)
    print(all_params_map)
    print(f"✅ 完成展示最优参数，花费时间：{time.time() - s_time:.2f}秒，累计时间：{(time.time() - r_time):.3f}秒")

    # 保存batch字典
    __save_batch_params(batch)
