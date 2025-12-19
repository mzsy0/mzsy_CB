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
# 因子的名称，可以是数据中有的，也可以是在data_process函数中计算出来的
main_factor = "factor_市值"
sub_factor = "factor_开盘至今涨幅_0945"

other_factor_list = [
    # 'factor_成交额缩量因子_(10,60)', 'factor_换手率_1'
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
    # df['成交额市值复合因子'] = df['factor_成交额缩量因子_(10,60)'] + df['factor_换手率_1']
    return df


"""
由于底层数据是1D级别的，所以数据量特别大，因子分析的计算量也比较大
为了减少内存开销，增加计算速度，因子分析默认只针对5_0周期进行分析
可以通过更改配置实现针对其他周期的计算，但不支持M系列的周期
"""


# endregion


def double_factor_analysis(main, sub, func, cfg, _other_factor_list, boost):
    # 因子分析需要用到的配置数据
    cfg.main = main if main.startswith("factor_") else f"factor_{main}"
    cfg.sub = sub if sub.startswith("factor_") else f"factor_{sub}"
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
        cfg.main,
        cfg.sub,
        "新版申万一级行业名称",
        "下周期涨跌幅",
        "下周期每天涨跌幅",
    ]

    start_time = datetime.datetime.now()

    # 读取因子数据
    factors_pkl = [
        _dir[:-4]
        for _dir in os.listdir(cfg.get_result_folder().parent.parent / "运行缓存/")
        if _dir.startswith("factor_")
    ]
    factor_list = []
    if cfg.main in factors_pkl:
        factor_list.append(cfg.main)

    if cfg.sub in factors_pkl:
        factor_list.append(cfg.sub)

    if _other_factor_list is not None:
        for _other_factor in _other_factor_list:
            _other_factor = _other_factor if _other_factor.startswith("factor_") else f"factor_{_other_factor}"
            if _other_factor in factors_pkl:
                factor_list.append(_other_factor)
            else:
                raise ValueError(f"{_other_factor} 因子名输入有误")

    factor_df = tf.get_data(cfg, factor_list, boost)

    # 存放图片的列表
    fig_list = []

    # ===计算双因子的热力图
    mix_nv, mix_prop, filter_nv_ms, filter_nv_sm = tf.get_group_nv_double(factor_df, cfg)

    # 画双因子平均收益组合热力图
    fig_list.append(
        pf.draw_hot_plotly(
            x=mix_nv.columns,
            y=mix_nv.index,
            z=mix_nv,
            title=f"双因子组合 - 日平均收益(‰)<br />主：{cfg.main}   次：{cfg.sub}",
        )
    )
    # 画双因子平均占比组合热力图
    fig_list.append(
        pf.draw_hot_plotly(
            x=mix_prop.columns,
            y=mix_prop.index,
            z=mix_prop,
            title=f"双因子组合 - 平均占比(%)<br />主：{cfg.main}   次：{cfg.sub}",
        )
    )
    # 画双因子平均收益过滤热力图
    fig_list.append(
        pf.draw_hot_plotly(
            x=filter_nv_ms.columns,
            y=filter_nv_ms.index,
            z=filter_nv_ms,
            title=f"双因子过滤 - 日平均收益(‰)<br />在【{cfg.main}】分组的基础上，对【{cfg.sub}】分组",
        )
    )
    fig_list.append(
        pf.draw_hot_plotly(
            x=filter_nv_sm.columns,
            y=filter_nv_sm.index,
            z=filter_nv_sm,
            title=f"双因子过滤 - 日平均收益(‰)<br />在【{cfg.sub}】分组的基础上，对【{cfg.main}】分组",
        )
    )

    # ===计算双因子风格暴露
    style_corr, corr_txt = tf.get_style_corr_double(factor_df, cfg)

    # 画双因子风格暴露图
    fig_list.append(
        pf.draw_three_bar_plotly(
            x=style_corr["风格"],
            y1=style_corr["相关系数_主因子"],
            y2=style_corr["相关系数_次因子"],
            y3=style_corr["相关系数_双因子"],
            title=corr_txt,
        )
    )

    start_date = factor_df["交易日期"].min().strftime("%Y/%m/%d")
    end_date = factor_df["交易日期"].max().strftime("%Y/%m/%d")
    title = f"分析区间：{start_date} - {end_date}  分析周期：{cfg.period_offset}"

    # # ===整合上面所有的图
    save_path = tf.get_folder_path(cfg.get_analysis_folder(), "双因子分析")
    pf.merge_html(
        save_path, fig_list=fig_list, strategy_file=f"{cfg.main}和{cfg.sub}_分析报告", bbs_id="45302", title=title
    )

    print(f"双因子分析完成，耗时：{datetime.datetime.now() - start_time}")


if __name__ == "__main__":
    print("开始运行因子分析程序...")
    conf = load_config()
    conf.bins = 10  # 设置分组数量
    conf.limit = 100  # 设置每周期最少需要多少个股票
    conf.fee_rate = (1 - conf.c_rate) * (1 - conf.c_rate - conf.t_rate)  # 提前计算好手粗费的比例
    conf.period_offset = "5_0"  # 只针对5_0周期进行分析
    double_factor_analysis(main_factor, sub_factor, data_process, conf, other_factor_list, boost=True)
