# -*- coding: utf-8 -*-
"""
邢不行｜策略分享会
股票量化策略框架𝓟𝓻𝓸

版权所有 ©️ 邢不行
微信: xbx1717

本代码仅供个人学习使用，未经授权不得复制、修改或用于商业用途。

Author: 邢不行
"""

import datetime
import os
import tools.utils.pfunctions as pf
import tools.utils.tfunctions as tf
from core.model.backtest_config import load_config

# region =====需要配置的内容=====
# 因子的名称，可以是数据中有的，按照运行缓存中的因子名输入，也可以是在data_process函数中计算出来的
factor_name = "市值"

# 输入其他需要的基础因子，可用于后续例如复合因子的计算，按照运行缓存中的因子名输入
other_factor_list = [
    # 'factor_成交额缩量因子_(10,60)', 'factor_市值_None'
]


def data_process(df):
    """
    在这个函数里面处理数据，主要是：过滤，计算符合因子等等
    :param df:
    :return:
    """

    # 案例1：增加分域的代码
    # df['总市值分位数'] = df.groupby('交易日期')['总市值'].rank(pct=True)
    # df = df[df['总市值分位数'] >= 0.9]
    # df = df[df['收盘价'] < 100]

    # 案例2：增加计算复合因子的代码
    # df['总市值排名'] = df.groupby('交易日期')['总市值'].rank()
    # df['成交额排名'] = df.groupby('交易日期')['成交额'].rank(ascending=False)
    # df['复合因子'] = df['总市值排名'] + df['成交额排名']

    # df['成交额市值复合因子'] = df['factor_成交额缩量因子_(10,60)'] + df['factor_市值_None']
    return df


"""
由于底层数据是1D级别的，所以数据量特别大，因子分析的计算量也比较大
为了减少内存开销，增加计算速度，因子分析默认只针对5_0周期进行分析
可以通过更改配置实现针对其他周期的计算，但不支持M系列的周期
"""


# endregion


def factor_analysis(name, func, cfg, _other_factor_list, boost):
    # 因子分析需要用到的配置数据
    cfg.fa_name = name if name.startswith("factor_") else f"factor_{name}"
    cfg.func = func
    cfg.keep_cols = [
        "交易日期",
        "股票代码",
        "股票名称",
        "下日_是否交易",
        "下日_开盘涨停",
        "下日_是否ST",
        "下日_是否退市",
        "上市至今交易天数",
        cfg.fa_name,
        "新版申万一级行业名称",
        "下周期涨跌幅",
        "下周期每天涨跌幅",
    ]
    cfg.ind_name_change = (
        {
            "采掘": "煤炭",
            "化工": "基础化工",
            "电气设备": "电力设备",
            "休闲服务": "社会服务",
            "纺织服装": "纺织服饰",
            "商业贸易": "商贸零售",
        },
    )

    start_time = datetime.datetime.now()

    # 读取因子数据
    factors_pkl = [
        _dir[:-4]
        for _dir in os.listdir(cfg.get_result_folder().parent.parent / "运行缓存/")
        if _dir.startswith("factor_")
    ]
    factor_list = []
    if cfg.fa_name in factors_pkl:
        factor_list.append(cfg.fa_name)
    if _other_factor_list is not None:
        for _other_factor in _other_factor_list:
            _other_factor = _other_factor if _other_factor.startswith("factor_") else f"factor_{_other_factor}"
            if _other_factor in factors_pkl:
                factor_list.append(_other_factor)
            else:
                raise ValueError(f"{_other_factor} 因子名输入有误")

    # 读取因子数据
    factor_df = tf.get_data(cfg, factor_list, boost)

    # 存放图片的列表
    fig_list = []

    # ===计算因子的IC
    ic, ic_info, ic_month = tf.get_ic(factor_df, cfg)
    # 添加ic的曲线图
    fig_list.append(
        pf.draw_ic_plotly(x=ic["交易日期"], y1=ic["RankIC"], y2=ic["累计RankIC"], title="因子RankIC图", info=ic_info)
    )
    # 添加阅读ic的热力图
    fig_list.append(
        pf.draw_hot_plotly(x=ic_month.columns, y=ic_month.index, z=ic_month, title="RankIC热力图(行：年份，列：月份)")
    )

    # ===计算因子的分组资金曲线及净值
    group_nv, group_value, group_hold_value = tf.get_group_net_value(factor_df, cfg)
    # 添加分组资金曲线图
    cols_list = [col for col in group_nv.columns if "第" in col]
    fig_list.append(
        pf.draw_line_plotly(
            x=group_nv["交易日期"], y1=group_nv[cols_list], y2=group_nv["多空净值"], if_log=True, title="分组资金曲线"
        )
    )
    # 添加分组净值图
    fig_list.append(pf.draw_bar_plotly(x=group_value["分组"], y=group_value["净值"], title="分组净值"))
    # 添加分组持仓走势
    fig_list.append(
        pf.draw_line_plotly(
            x=group_hold_value["时间"],
            y1=group_hold_value[cols_list],
            update_xticks=True,
            if_log=False,
            title="分组持仓走势",
        )
    )

    # ===计算因子的风格暴露
    style_corr = tf.get_style_corr(factor_df, cfg)
    # 添加风格暴露图
    fig_list.append(
        pf.draw_bar_plotly(x=style_corr["风格"], y=style_corr["相关系数"], title="因子风格暴露图", y_range=[-1.0, 1.0])
    )

    # ===计算行业平均IC以及行业占比
    industry_df = tf.get_class_ic_and_pct(factor_df, cfg)
    # 添加行业平均IC
    fig_list.append(
        pf.draw_bar_plotly(x=industry_df["新版申万一级行业名称"], y=industry_df["RankIC"], title="行业RankIC图")
    )
    # 添加行业占比图
    fig_list.append(
        pf.draw_double_bar_plotly(
            x=industry_df["新版申万一级行业名称"],
            y1=industry_df["因子第一组选股在各行业的占比"],
            y2=industry_df["因子最后一组选股在各行业的占比"],
            title="行业占比（可能会受到行业股票数量的影响）",
        )
    )

    # ===计算不同市值分组内的平均IC以及市值占比
    market_df = tf.get_class_ic_and_pct(factor_df, cfg, is_industry=False)
    # 添加市值分组平均IC
    fig_list.append(pf.draw_bar_plotly(x=market_df["市值分组"], y=market_df["RankIC"], title="市值分组RankIC"))
    # 添加市值分组占比图
    info = "1-{bins}代表市值从小到大分{bins}组".format(bins=cfg.bins)
    fig_list.append(
        pf.draw_double_bar_plotly(
            x=market_df["市值分组"],
            y1=market_df["因子第一组选股在各市值分组的占比"],
            y2=market_df["因子最后一组选股在各市值分组的占比"],
            title="市值占比",
            info=info,
        )
    )

    # ===计算因子得分
    score = tf.get_factor_score(ic, group_value)
    start_date = factor_df["交易日期"].min().strftime("%Y/%m/%d")
    end_date = factor_df["交易日期"].max().strftime("%Y/%m/%d")

    title = f"{cfg.fa_name} 分析区间：{start_date} - {end_date}  分析周期：{cfg.period_offset}  因子得分：{score:.2f}"

    # ===整合上面所有的图
    save_path = tf.get_folder_path(cfg.get_analysis_folder(), "单因子分析")
    pf.merge_html(save_path, fig_list=fig_list, strategy_file=f"{cfg.fa_name}因子分析报告", bbs_id="31614", title=title)
    print(f"汇总数据并画图完成，耗时：{datetime.datetime.now() - start_time}")
    print(f"{cfg.fa_name} 因子分析完成，耗时：{datetime.datetime.now() - start_time}")


if __name__ == "__main__":
    print("开始运行因子分析程序...")
    conf = load_config()
    conf.bins = 10  # 设置分组数量
    conf.limit = 100  # 设置每周期最少需要多少个股票
    conf.fee_rate = (1 - conf.c_rate) * (1 - conf.c_rate - conf.t_rate)  # 提前计算好手续费的比例
    conf.period_offset = "5_0"  # 只针对5_0周期进行分析
    factor_analysis(factor_name, data_process, conf, other_factor_list, boost=True)
