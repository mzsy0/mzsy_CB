# -*- coding: utf-8 -*-
"""
邢不行｜策略分享会
股票量化策略框架𝓟𝓻𝓸

版权所有 ©️ 邢不行
微信: xbx1717

本代码仅供个人学习使用，未经授权不得复制、修改或用于商业用途。

Author: 邢不行
"""

import itertools
import operator
import os
import warnings
from functools import reduce
from pathlib import Path
import pandas as pd

import tools.utils.pfunctions as pf
from core.utils.path_kit import get_folder_path

warnings.filterwarnings("ignore")


# ====== 公共函数 ======
def dict_itertools(dict_):
    keys = list(dict_.keys())
    values = list(dict_.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def filter_dataframe(df, filter_dict):
    conditions = [df[col].isin(values) for col, values in filter_dict.items()]
    return df[reduce(operator.and_, conditions)] if conditions else df.copy()


def prepare_data():
    """生成参数组合并过滤"""
    params_df = pd.DataFrame(dict_itertools(batch))
    try:
        params_df["参数组合"] = sorted(
            [x for x in result_folder_path.iterdir() if x.is_dir()], key=lambda x: int(x.stem.split("-")[0][1:])
        )
    except ValueError:
        print("请检查trav_name是否有误")
        exit()
    return filter_dataframe(params_df, limit_dict)


def load_and_process_data(df_left, result_dir: Path):
    """加载并处理策略评价数据"""
    if evaluation_indicator not in [
        "累积净值",
        "年化收益",
        "最大回撤",
        "年化收益/回撤比",
        "盈利周期数",
        "亏损周期数",
        "胜率",
        "每周期平均收益",
        "盈亏收益比",
        "单周期最大盈利",
        "单周期大亏损",
        "最大连续盈利周期数",
        "最大连续亏损周期数",
        "收益率标准差",
    ]:
        raise ValueError("评价指标有误，按要求输入")

    if evaluation_indicator == "年化收益":
        time_list = []
        for folder in df_left["参数组合"]:
            # 读取策略评价数据
            stats_path = result_dir / folder / "策略评价.csv"
            stats_temp = pd.read_csv(stats_path, encoding="utf-8")
            stats_temp.columns = ["evaluation_indicator", "value"]
            if stats_temp.empty:
                raise ValueError(f"{folder} 文件夹内策略评价数据为空，请检查数据")
            stats_temp = stats_temp.set_index("evaluation_indicator")
            df_left.loc[df_left["参数组合"] == folder, "all"] = stats_temp.loc[evaluation_indicator, "value"]

            # 读取年度数据
            years_path = result_dir / folder / "年度账户收益.csv"
            years_return = pd.read_csv(years_path, encoding="utf-8")
            if years_return.empty:
                raise ValueError(f"{folder} 文件夹内年度账户收益数据为空，请检查数据")
            time_list = list(years_return["交易日期"].sort_values(ascending=False))
            for time in time_list:
                df_left.loc[df_left["参数组合"] == folder, time] = years_return.loc[
                    years_return["交易日期"] == time, "涨跌幅"
                ].iloc[0]

        # 格式转换
        df_left[["all"] + time_list] = df_left[["all"] + time_list].map(
            lambda x: float(x.replace("%", "")) / 100 if "%" in str(x) else float(x)
        )
        return time_list
    else:
        for folder in df_left["参数组合"]:
            stats_path = result_dir / folder / "策略评价.csv"
            stats_temp = pd.read_csv(stats_path, encoding="utf-8")
            if stats_temp.empty:
                raise ValueError(f"{folder} 文件夹内策略评价数据为空，请检查数据")
            stats_temp.columns = ["evaluation_indicator", "value"]
            stats_temp = stats_temp.set_index("evaluation_indicator")
            df_left.loc[df_left["参数组合"] == folder, evaluation_indicator] = stats_temp.loc[
                evaluation_indicator, "value"
            ]

        df_left[evaluation_indicator] = df_left[evaluation_indicator].apply(
            lambda x: float(x.replace("%", "")) / 100 if "%" in str(x) else float(x)
        )
        return None


def generate_plots(df_left, params, output_dir: Path, analysis_type, time_list):
    """根据分析类型生成图表"""
    fig_list = []
    html_name = f"年化收益_回撤比.html" if evaluation_indicator == "年化收益/回撤比" else f"{evaluation_indicator}.html"

    if "hold_period" in df_left.columns:
        df_left["periods"] = df_left["hold_period"].apply(lambda x: int(x[:-1]))
        df_left = df_left.sort_values(by=["periods"])

    if analysis_type == "double":
        x_, y_ = params

        if evaluation_indicator == "年化收益":
            for time in ["all"] + time_list:
                temp = pd.pivot_table(df_left, index=y_, columns=x_, values=time)
                fig = pf.draw_params_heatmap_plotly(temp, title=time)
                fig_list.append(fig)
        else:
            temp = pd.pivot_table(df_left, index=y_, columns=x_, values=evaluation_indicator)
            fig = pf.draw_params_heatmap_plotly(temp, title=evaluation_indicator)
            fig_list.append(fig)
        html_name = f"{x_}_{y_}_{html_name}"

    else:
        param = params
        if evaluation_indicator == "年化收益":
            sub_df = df_left[[param] + ["all"] + time_list].copy()
            sub_df[param] = sub_df[param].map(lambda x: f"{param}_{x}")
            sub_df = sub_df.set_index(param)
            fig = pf.draw_params_bar_plotly(sub_df, evaluation_indicator)
        else:
            x_axis = df_left[param].map(lambda x: f"{param}_{x}")
            fig = pf.draw_bar_plotly(
                x_axis, df_left[evaluation_indicator], title=evaluation_indicator, pic_size=[1800, 600]
            )
        fig_list.append(fig)
        html_name = f"{param}_{html_name}"

    if fig_list:
        title = "参数热力图" if analysis_type == "double" else "参数平原图"
        pf.merge_html_flexible(fig_list, output_dir / html_name, title=title)


# ====== 主逻辑 ======
def analyze_params(analysis_type):
    """参数分析主函数"""
    df_left = prepare_data()

    # 配置输出路径
    if analysis_type == "double":
        output_dir = out_folder_path / "参数热力图" / trav_name
        params = [param_x, param_y]
    else:
        output_dir = out_folder_path / "参数平原图" / trav_name
        params = param_x
    os.makedirs(output_dir, exist_ok=True)

    # 处理数据
    time_list = load_and_process_data(df_left, result_folder_path)

    # 生成图表
    generate_plots(df_left, params, output_dir, analysis_type, time_list)


if __name__ == "__main__":
    # ====== 配置信息 ======
    trav_name = "选股策略混合"  # 用于读取 data/遍历结果/ 中的遍历回测结果
    # 回测路径和参数分析输出路径
    result_folder_path = get_folder_path("data", "遍历结果", trav_name, auto_create=False)
    out_folder_path = get_folder_path("data", "分析结果", "参数分析")

    # 参数设置
    batch = {
        "params": [0.5, 0.6],
        "limit": [200, 500],
    }

    # 若绘制单参数平原图，param_x 填写变量，param_y=''
    # 若绘制双参数热力图，则 param_x和param_y 填写变量, param_为热力图x轴变量，param_y为热力图y轴变量，可按需更改
    param_x = "params"
    param_y = "limit"

    # 这里需要固定非观测参数，然后画参数图，例如该案例固定hold_period== 12H，来看LowPrice和QuoteVolumeMean的参数热力图
    # 注意点：多参数画图，必须固定其他参数。单参数平原需固定该参数以外的其他参数，双参数热力图需固定除两参数以外的参数
    limit_dict = {
        # 'rebalance_time': ["0935-0945"],
        # "select_num": [5],
        # '换手率': [20],
        # '换手率': [20],
    }

    # 分析指标，支持以下：
    # 累积净值、年化收益、最大回撤、年化收益/回撤比、盈利周期数、亏损周期数、胜率、每周期平均收益
    # 盈亏收益比、单周期最大盈利、单周期大亏损、最大连续盈利周期数、最大连续亏损周期数、收益率标准差
    evaluation_indicator = "年化收益"

    # ====== 主逻辑 ======
    # 进行参数分析
    analysis_type = "single" if len(param_y.strip()) == 0 else "double"
    analyze_params(analysis_type)
