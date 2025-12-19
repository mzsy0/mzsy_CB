import pandas as pd
import os
from datetime import datetime

start_date = "2025/10/10"
end_date = "2025/10/23"

date_str = datetime.today().strftime("%Y%m%d")

file1_path = r"C:\Users\29856\Desktop\VAL-RuiDaInternatio-Summary.20251021(1).xlsx"
file2_path = r"C:\Users\29856\Desktop\China Convertible Arbitrage Fund I_Dummy Master_ValuationReport_20251023.xlsx"
file3_path = r"C:\Users\29856\Desktop\China Convertible Arbitrage Fund I_Dummy Master_ValuationReport_20251009.xlsx"
file4_path = r"C:\Users\29856\Desktop\stock_bond_Summary.xlsx"
file5_path = r"C:\Users\29856\Desktop\MGN-RuiDaInternatio-2585922_2585924.20251021.xlsx"
output_path = f"CICC&CITIC香港单标的估值表计算_{date_str}.xlsx"

def delete_stock_CITIC(input_file,MGN_file):

    results_sum = {}

    try:
        # 读取Excel文件
        print("读取Excel文件...")
        df = pd.read_excel(input_file, sheet_name='Position valuation report', header=4)
        
        # 检查列名
        print("表格列名:", df.columns.tolist())
        
        # 显示原始数据行数
        original_count = len(df)
        print(f"原始数据行数: {original_count}")
        
        # 创建要删除的股票代码的模式列表（添加" CH"后缀）
        patterns_to_remove = [f"{stock} CH" for stock in stocks_to_remove]
        print(f"要删除的股票模式: {patterns_to_remove[:5]}...")  # 只显示前5个
        
        # 过滤数据 - 保留不匹配要删除股票代码的行
        filtered_df = df[~df['Underlying Security Name'].isin(patterns_to_remove)]
        
        # 显示过滤后的数据行数
        filtered_count = len(filtered_df)
        print(f"过滤后数据行数: {filtered_count}")
        print(f"删除了 {original_count - filtered_count} 行数据")
        
        # 显示被删除的股票代码
        removed_stocks = df[df['Underlying Security Name'].isin(patterns_to_remove)]['Underlying Security Name'].unique()
        print(f"被删除的股票代码: {list(removed_stocks)}")

        CITIC_sum = filtered_df['Total Position P&L \n(Settle ccy)'].sum()

        df_MGN = pd.read_excel(MGN_file, sheet_name='Margin Summary', header=44)

        results_sum['中信账户总盈亏']=CITIC_sum
        results_sum['其中：券息利息']=df_MGN['USD'].iloc[0]

        return filtered_df
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None
    
stocks_to_remove = [
        "000001", "000651", "000858", "002352", "002371", 
        "002475", "002594", "002714", "300059", "300308", 
        "300502", "300750", "300760", "600000", "600028", 
        "600036", "600276", "600519", "600690", "600809", 
        "600900", "600919", "601088", "601138", "601166", 
        "601211", "601225", "601288", "601318", "601319", 
        "601328", "601398", "601601", "601658", "601668", 
        "601766", "601816", "601818", "601857", "601899", 
        "601919", "601939", "601988", "601998", "603259", 
        "603288", "688041", "688235", "688256"
    ]






def process_stock_data(a, b, start_date, end_date, file1, file2, file3, output_file):
    """
    处理三个Excel表格，计算7个变量并输出到新Excel文件
    
    参数:
    a, b: 输入的股票代码
    start_date, end_date: 开始和结束日期
    file1, file2, file3: 三个Excel文件路径
    output_file: 输出文件路径
    """
    
    # 存储结果的字典
    results = {}
    results['Effective Date'] = end_date
    results['股票代码'] = stock_a
    results['转债代码'] = stock_b
    
    try:
        # 处理第一个表格 - 使用新的文件格式
        print("处理第一个表格...")
        df1 = pd.read_excel(file1, sheet_name='Position valuation report', header=4)
        
        # 检查列名
        print("第一个表格的列名:", df1.columns.tolist())
        
        # 尝试多种匹配方式 - 匹配输入a
        a_rows = pd.DataFrame()  # 空DataFrame

        # 方式1: 匹配Underlying Security Name
        if 'Underlying Security Name' in df1.columns:
            # 尝试多种匹配模式
            patterns = [
                f"{a} CH",  # 匹配以a开头，后面有空格和CH
                f"{a}CH",   # 匹配以a开头，后面有CH
                str(a),     # 直接匹配
            ]
            
            for pattern in patterns:
                temp_rows = df1[df1['Underlying Security Name'].astype(str).str.startswith(pattern, na=False)]
                if not temp_rows.empty:
                    print(f"使用模式 '{pattern}' 在Underlying Security Name中找到匹配{a}的行数: {len(temp_rows)}")
                    a_rows = temp_rows
                    break
            
            # 如果还没找到，尝试包含匹配
            if a_rows.empty:
                a_rows = df1[df1['Underlying Security Name'].astype(str).str.contains(str(a), na=False)]
                print(f"包含匹配 - 在Underlying Security Name中找到包含{a}的行数: {len(a_rows)}")

        # 如果还没找到，尝试其他列
        if a_rows.empty and 'RIC' in df1.columns:
            patterns = [
                f"{a}.SZ",  # 匹配以a开头，后面有.SZ
                f"{a}.SH",  # 匹配以a开头，后面有.SH
                str(a),     # 直接匹配
            ]
            
            for pattern in patterns:
                temp_rows = df1[df1['RIC'].astype(str).str.startswith(pattern, na=False)]
                if not temp_rows.empty:
                    print(f"使用模式 '{pattern}' 在RIC列中找到匹配{a}的行数: {len(temp_rows)}")
                    a_rows = temp_rows
                    break
        
        if a_rows.empty and 'Local Code' in df1.columns:
            patterns = [
                f"{a} CS",  # 匹配以a开头，后面有空格和CS
                f"{a}CS",   # 匹配以a开头，后面有CS
                str(a),     # 直接匹配
            ]
            
            for pattern in patterns:
                temp_rows = df1[df1['Local Code'].astype(str).str.startswith(pattern, na=False)]
                if not temp_rows.empty:
                    print(f"使用模式 '{pattern}' 在Local Code列中找到匹配{a}的行数: {len(temp_rows)}")
                    a_rows = temp_rows
                    break

        if not a_rows.empty:
            print("\n匹配到的数据详情:")
            # 显示相关的列以便调试
            display_columns = []
            for col in ['Underlying Security Name', 'RIC', 'Local Code', 'Security Description', 'Total Position P&L \n(Settle ccy)']:
                if col in a_rows.columns:
                    display_columns.append(col)
            
            if display_columns:
                print(a_rows[display_columns].head())
        else:
            print(f"\n未找到任何包含 '{a}' 的记录")
            # 打印前几个存在的代码供参考
            if 'Underlying Security Name' in df1.columns:
                all_codes = df1['Underlying Security Name'].dropna().astype(str).unique()
                print(f"第一个表格中存在的代码示例: {all_codes[:10]}")
        
        # 计算Total Position P&L (Settle ccy)总和
        if not a_rows.empty and 'Total Position P&L \n(Settle ccy)' in df1.columns:
            zz_summary = a_rows['Total Position P&L \n(Settle ccy)'].sum()
            print(f"找到的Total Position P&L值: {a_rows['Total Position P&L \n(Settle ccy)'].tolist()}")
        else:
            zz_summary = 0
            print("警告: 未找到匹配的行或缺少'Total Position P&L (Settle ccy)'列")

        if not a_rows.empty and 'Current Accrual \n(Settle ccy)' in df1.columns:
            Summary_coupons = a_rows['Current Accrual \n(Settle ccy)'].sum()
            print(f"找到的Current Accrual \n(Settle ccy)值: {a_rows['Current Accrual \n(Settle ccy)'].tolist()}")
        else:
            Summary_coupons = 0
            print("警告: 未找到匹配的行或缺少'Current Accrual \n(Settle ccy)'列")

        # 处理第二个表格 - Summary
        print("\n处理第二个表格的Summary...")
        df2_summary = pd.read_excel(file2, sheet_name='Summary')
        
        # 检查列名
        print("第二个表格Summary的列名:", df2_summary.columns.tolist())
        
        # 根据调试信息，第二个表格的Ticker格式是"001212.SZ"，我们需要添加后缀
        # 尝试多种后缀格式
        a_patterns = [
            f"{a}.SZ",  # 深圳交易所
            f"{a}.SH",  # 上海交易所
            f"{a}.BJ",  # 北京交易所
        ]
        
        b_patterns = [
            f"{b}.SZ",  # 深圳交易所
            f"{b}.SH",  # 上海交易所
            f"{b}.BJ",  # 北京交易所
        ]
        
        # 匹配Ticker列
        a_rows_2 = pd.DataFrame()
        b_rows_2 = pd.DataFrame()
        
        for pattern in a_patterns:
            temp_rows = df2_summary[df2_summary['Ticker'].astype(str).str.startswith(pattern, na=False)]
            if not temp_rows.empty:
                print(f"使用模式 '{pattern}' 在第二个表格中找到匹配{a}的行数: {len(temp_rows)}")
                a_rows_2 = temp_rows
                break
        
        for pattern in b_patterns:
            temp_rows = df2_summary[df2_summary['Ticker'].astype(str).str.startswith(pattern, na=False)]
            if not temp_rows.empty:
                print(f"使用模式 '{pattern}' 在第二个表格中找到匹配{b}的行数: {len(temp_rows)}")
                b_rows_2 = temp_rows
                break
        
        # 如果没找到，尝试包含匹配
        if a_rows_2.empty:
            a_rows_2 = df2_summary[df2_summary['Ticker'].astype(str).str.contains(str(a), na=False)]
            print(f"包含匹配 - 在第二个表格中找到包含{a}的行数: {len(a_rows_2)}")
        
        if b_rows_2.empty:
            b_rows_2 = df2_summary[df2_summary['Ticker'].astype(str).str.contains(str(b), na=False)]
            print(f"包含匹配 - 在第二个表格中找到包含{b}的行数: {len(b_rows_2)}")
        
        if not a_rows_2.empty:
            print(f"匹配到的{a}的Ticker: {a_rows_2['Ticker'].tolist()}")
        else:
            print(f"未找到任何包含 '{a}' 的记录")
            
        if not b_rows_2.empty:
            print(f"匹配到的{b}的Ticker: {b_rows_2['Ticker'].tolist()}")
        else:
            print(f"未找到任何包含 '{b}' 的记录")
        
        # 计算Total SWAP Value (Settlement CCY)总和
        if not a_rows_2.empty and 'Total SWAP Value (Settlement CCY)' in df2_summary.columns:
            a_valuation_start = a_rows_2['Total SWAP Value (Settlement CCY)'].sum()
            print(f"找到的{a}的Total SWAP Value: {a_rows_2['Total SWAP Value (Settlement CCY)'].tolist()}")
        else:
            a_valuation_start = 0
            print("警告: 未找到匹配的行或缺少'Total SWAP Value (Settlement CCY)'列")
            
        if not b_rows_2.empty and 'Total SWAP Value (Settlement CCY)' in df2_summary.columns:
            b_valuation_ending = b_rows_2['Total SWAP Value (Settlement CCY)'].sum()
            print(f"找到的{b}的Total SWAP Value: {b_rows_2['Total SWAP Value (Settlement CCY)'].tolist()}")
        else:
            b_valuation_ending = 0
            print("警告: 未找到匹配的行或缺少'Total SWAP Value (Settlement CCY)'列")
        
        if not a_rows_2.empty and 'Interest Accrual in Settlement CCY' in df2_summary.columns:
            Valuation_stock_coupons = a_rows_2['Interest Accrual in Settlement CCY'].sum()
            print(f"找到的{a}的Interest Accrual in Settlement CCY: {a_rows_2['Interest Accrual in Settlement CCY'].tolist()}")
        else:
            Valuation_stock_coupons = 0
            print("警告: 未找到匹配的行或缺少'Interest Accrual in Settlement CCY'列")
            
        if not b_rows_2.empty and 'Interest Accrual in Settlement CCY' in df2_summary.columns:
            Valuation_bond_coupons = b_rows_2['Interest Accrual in Settlement CCY'].sum()
            print(f"找到的{b}的Interest Accrual in Settlement CCY: {b_rows_2['Interest Accrual in Settlement CCY'].tolist()}")
        else:
            Valuation_bond_coupons = 0
            print("警告: 未找到匹配的行或缺少'Interest Accrual in Settlement CCY'列")

        # 处理第三个表格 - Summary (与第二个表格相同的操作)
        print("\n处理第三个表格的Summary...")
        df3_summary = pd.read_excel(file3, sheet_name='Summary')
        
        # 检查列名
        print("第三个表格Summary的列名:", df3_summary.columns.tolist())
        
        # 匹配Ticker列（使用与第二个表格相同的格式）
        a_rows_3 = pd.DataFrame()
        b_rows_3 = pd.DataFrame()
        
        for pattern in a_patterns:
            temp_rows = df3_summary[df3_summary['Ticker'].astype(str).str.startswith(pattern, na=False)]
            if not temp_rows.empty:
                print(f"使用模式 '{pattern}' 在第三个表格中找到匹配{a}的行数: {len(temp_rows)}")
                a_rows_3 = temp_rows
                break
        
        for pattern in b_patterns:
            temp_rows = df3_summary[df3_summary['Ticker'].astype(str).str.startswith(pattern, na=False)]
            if not temp_rows.empty:
                print(f"使用模式 '{pattern}' 在第三个表格中找到匹配{b}的行数: {len(temp_rows)}")
                b_rows_3 = temp_rows
                break
        
        # 如果没找到，尝试包含匹配
        if a_rows_3.empty:
            a_rows_3 = df3_summary[df3_summary['Ticker'].astype(str).str.contains(str(a), na=False)]
            print(f"包含匹配 - 在第三个表格中找到包含{a}的行数: {len(a_rows_3)}")
        
        if b_rows_3.empty:
            b_rows_3 = df3_summary[df3_summary['Ticker'].astype(str).str.contains(str(b), na=False)]
            print(f"包含匹配 - 在第三个表格中找到包含{b}的行数: {len(b_rows_3)}")
        
        if not a_rows_3.empty:
            print(f"匹配到的{a}的Ticker: {a_rows_3['Ticker'].tolist()}")
        else:
            print(f"未找到任何包含 '{a}' 的记录")
            
        if not b_rows_3.empty:
            print(f"匹配到的{b}的Ticker: {b_rows_3['Ticker'].tolist()}")
        else:
            print(f"未找到任何包含 '{b}' 的记录")
        
        # 计算Total SWAP Value (Settlement CCY)总和
        if not a_rows_3.empty and 'Total SWAP Value (Settlement CCY)' in df3_summary.columns:
            a2_valuation_start = a_rows_3['Total SWAP Value (Settlement CCY)'].sum()
        else:
            a2_valuation_start = 0
            print("警告: 未找到匹配的行或缺少'Total SWAP Value (Settlement CCY)'列")
            
        if not b_rows_3.empty and 'Total SWAP Value (Settlement CCY)' in df3_summary.columns:
            b2_valuation_ending = b_rows_3['Total SWAP Value (Settlement CCY)'].sum()
        else:
            b2_valuation_ending = 0
            print("警告: 未找到匹配的行或缺少'Total SWAP Value (Settlement CCY)'列")

        if not a_rows_3.empty and 'Interest Accrual in Settlement CCY' in df3_summary.columns:
            Valuation_stock2_coupons = a_rows_3['Interest Accrual in Settlement CCY'].sum()
        else:
            Valuation_stock2_coupons = 0
            print("警告: 未找到匹配的行或缺少'Interest Accrual in Settlement CCY'列")
            
        if not b_rows_3.empty and 'Interest Accrual in Settlement CCY' in df3_summary.columns:
            Valuation_bond2_coupons = b_rows_3['Interest Accrual in Settlement CCY'].sum()
        else:
            Valuation_bond2_coupons = 0
            print("警告: 未找到匹配的行或缺少'Interest Accrual in Settlement CCY'列")
        
        # 处理第二个表格 - EQ Unwound
        print("\n处理第二个表格的EQ Unwound...")
        df2_eq_unwound = pd.read_excel(file2, sheet_name='EQ Unwound')
        
        # 检查列名
        print("第二个表格EQ Unwound的列名:", df2_eq_unwound.columns.tolist())
        
        # 将Settle Date列转换为datetime格式以便比较
        df2_eq_unwound['Settle Date'] = pd.to_datetime(df2_eq_unwound['Settle Date'], errors='coerce')
        
        # 将开始和结束日期转换为datetime格式
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        print(f"筛选日期范围: {start_dt} 到 {end_dt}")
        
        # 筛选日期范围内的数据
        date_filtered_data = df2_eq_unwound[
            (df2_eq_unwound['Settle Date'] >= start_dt) & 
            (df2_eq_unwound['Settle Date'] <= end_dt)
        ]
        
        print(f"在日期范围内找到的总行数: {len(date_filtered_data)}")
        
        # 根据调试信息，Security Code的格式可能是"002921.SZ"这样的格式
        # 尝试多种后缀格式
        a_security_patterns = [
            f"{a}.SZ",  # 深圳交易所
            f"{a}.SH",  # 上海交易所
            f"{a}.BJ",  # 北京交易所
        ]
        
        b_security_patterns = [
            f"{b}.SZ",  # 深圳交易所
            f"{b}.SH",  # 上海交易所
            f"{b}.BJ",  # 北京交易所
        ]
        
        # 在日期范围内匹配Security Code
        a_rows_eq = pd.DataFrame()
        b_rows_eq = pd.DataFrame()
        
        for pattern in a_security_patterns:
            temp_rows = date_filtered_data[date_filtered_data['Security Code'].astype(str).str.startswith(pattern, na=False)]
            if not temp_rows.empty:
                print(f"使用模式 '{pattern}' 在日期范围内找到匹配{a}的行数: {len(temp_rows)}")
                a_rows_eq = temp_rows
                break
        
        for pattern in b_security_patterns:
            temp_rows = date_filtered_data[date_filtered_data['Security Code'].astype(str).str.startswith(pattern, na=False)]
            if not temp_rows.empty:
                print(f"使用模式 '{pattern}' 在日期范围内找到匹配{b}的行数: {len(temp_rows)}")
                b_rows_eq = temp_rows
                break
        
        # 如果直接匹配没找到，尝试包含匹配
        if a_rows_eq.empty:
            a_rows_eq = date_filtered_data[date_filtered_data['Security Code'].astype(str).str.contains(str(a), na=False)]
            print(f"包含匹配 - 在日期范围内找到包含{a}的行数: {len(a_rows_eq)}")
        
        if b_rows_eq.empty:
            b_rows_eq = date_filtered_data[date_filtered_data['Security Code'].astype(str).str.contains(str(b), na=False)]
            print(f"包含匹配 - 在日期范围内找到包含{b}的行数: {len(b_rows_eq)}")
        
        if not a_rows_eq.empty:
            print(f"匹配到的{a}的Security Code和Total SWAP Value:")
            for idx, row in a_rows_eq.iterrows():
                print(f"  Security Code: {row['Security Code']}, Settle Date: {row['Settle Date']}, Total SWAP Value: {row['Total SWAP Value(Settle CCY)']}")
        
        if not b_rows_eq.empty:
            print(f"匹配到的{b}的Security Code和Total SWAP Value:")
            for idx, row in b_rows_eq.iterrows():
                print(f"  Security Code: {row['Security Code']}, Settle Date: {row['Settle Date']}, Total SWAP Value: {row['Total SWAP Value(Settle CCY)']}")
        
        # 计算Total SWAP Value(Settle CCY)总和
        if not a_rows_eq.empty and 'Total SWAP Value(Settle CCY)' in df2_eq_unwound.columns:
            data_a_sum = a_rows_eq['Total SWAP Value(Settle CCY)'].sum()
        else:
            data_a_sum = 0
            print("警告: 未找到匹配的行或缺少'Total SWAP Value(Settle CCY)'列")
            
        if not b_rows_eq.empty and 'Total SWAP Value(Settle CCY)' in df2_eq_unwound.columns:
            data_b_sum = b_rows_eq['Total SWAP Value(Settle CCY)'].sum()
            print(f"计算得到的data_b_sum: {data_b_sum}")
        else:
            data_b_sum = 0
            print("警告: 未找到匹配的行或缺少'Total SWAP Value(Settle CCY)'列")

        if not a_rows_eq.empty and 'Interest(Settle CCY)' in df2_eq_unwound.columns:
            eq_coupons_stock = a_rows_eq['Interest(Settle CCY)'].sum()
        else:
            eq_coupons_stock = 0
            print("警告: 未找到匹配的行或缺少'Interest(Settle CCY)'列")
            
        if not b_rows_eq.empty and 'Interest(Settle CCY)' in df2_eq_unwound.columns:
            eq_coupons_bond = b_rows_eq['Interest(Settle CCY)'].sum()
            print(f"计算得到的eq_coupons_bond: {eq_coupons_bond}")
        else:
            eq_coupons_bond = 0
            print("警告: 未找到匹配的行或缺少'Interest(Settle CCY)'列")

        results['组合总收益（含全部成本）'] = data_a_sum + a_valuation_start - a2_valuation_start + data_b_sum + b_valuation_ending - b2_valuation_ending + zz_summary
        print(f"组合总收益（含全部成本）:{data_a_sum + a_valuation_start - a2_valuation_start + data_b_sum + b_valuation_ending - b2_valuation_ending + zz_summary}")

        results['转债多头利息'] = eq_coupons_bond + Valuation_bond_coupons - Valuation_bond2_coupons
        print(f"转债多头利息:{eq_coupons_bond + Valuation_bond_coupons - Valuation_bond2_coupons}")

        results['股票利息券息'] = eq_coupons_stock + Valuation_stock_coupons - Valuation_stock2_coupons + Summary_coupons
        print(f"股票利息券息:{eq_coupons_stock + Valuation_stock_coupons - Valuation_stock2_coupons + Summary_coupons}")

        results['total position P&L'] = zz_summary
        print(f"total position P&L: {zz_summary}")

        results['CITIC current accrual'] = Summary_coupons
        print(f"CITIC current accrual: {Summary_coupons}")

        results['中金股票浮盈期末'] = a_valuation_start
        results['中金转债浮盈期末'] = b_valuation_ending
        print(f"中金股票浮盈期末: {a_valuation_start}")
        print(f"中金转债浮盈期末: {b_valuation_ending}")

        results['中金股票浮盈期末_券息'] = Valuation_stock_coupons
        results['中金转债浮盈期末_券息'] = Valuation_bond_coupons
        print(f"中金股票浮盈期末_券息: {Valuation_stock_coupons}")
        print(f"中金转债浮盈期末_券息: {Valuation_bond_coupons}")

        results['中金股票浮盈期初'] = a2_valuation_start
        results['中金转债浮盈期初'] = b2_valuation_ending
        print(f"中金股票浮盈期初: {a2_valuation_start}")
        print(f"中金转债浮盈期初: {b2_valuation_ending}")

        results['中金股票浮盈期初_券息'] = Valuation_stock2_coupons
        results['中金转债浮盈期初_券息'] = Valuation_bond2_coupons
        print(f"中金股票浮盈期初_券息: {Valuation_stock2_coupons}")
        print(f"中金转债浮盈期初_券息: {Valuation_bond2_coupons}")

        results['中金已实现股票'] = data_a_sum
        results['中金已实现转债'] = data_b_sum
        print(f"中金已实现股票: {data_a_sum}")
        print(f"中金已实现转债: {data_b_sum}")
        
        results['中金已实现股票_券息'] = eq_coupons_stock
        results['中金已实现转债_券息'] = eq_coupons_bond
        print(f"中金已实现股票_券息: {eq_coupons_stock}")
        print(f"中金已实现转债_券息: {eq_coupons_bond}")
        
        results['转债价差收益（含利息）'] = data_b_sum + b_valuation_ending - b2_valuation_ending
        print(f"中金已实现股票:{data_b_sum + b_valuation_ending - b2_valuation_ending}")

        results['股票价差收益（含利息券息）'] = data_a_sum + a_valuation_start - a2_valuation_start + zz_summary
        print(f"股票价差收益（含利息券息）:{data_a_sum + a_valuation_start - a2_valuation_start + zz_summary}")
        
        # 创建结果DataFrame并保存
        result_df = pd.DataFrame([results])
        
        # 检查目标Excel文件是否已存在
        if os.path.exists(output_file):
            # 若存在，先读取原文件数据
            old_df = pd.read_excel(output_file)
            # 将新结果与原数据合并（忽略原有索引，重新生成索引）
            combined_df = pd.concat([old_df, result_df], ignore_index=True)
            # 保存合并后的数据（覆盖原文件，实现"累加"效果）
            combined_df.to_excel(output_file, index=False)
        else:
            # 若文件不存在，直接保存新结果（首次创建）
            result_df.to_excel(output_file, index=False)

        print(f"\n结果已累加到: {output_file}")
        
        return results
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None

# 使用示例
def main_file(stock_a,stock_b,end_date):
    # 检查文件是否存在
    for file_path in [file1_path, file2_path, file3_path]:
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在 - {file_path}")
    
    # 执行处理
    results = process_stock_data(
        a=stock_a,
        b=stock_b,
        start_date=start_date,
        end_date=end_date,
        file1=file1_path,
        file2=file2_path,
        file3=file3_path,
        output_file=output_path
    )
    
    if results:
        print("\n处理完成！所有变量值:")
        for var_name, value in results.items():
            print(f"{var_name}: {value}")

if __name__ == "__main__":
    # df_input = pd.read_excel(file4_path, sheet_name='Sheet1', header=0)

    delete_stock_CITIC(input_file=file1_path,MGN_file=file5_path)
    
    # # 1. 初始索引设为“最后一行的索引”（总行数-1，避免越界）
    # i = len(df_input) - 1  
    # # 2. 循环条件：i >= 0（从最后一行遍历到第0行）
    # while i >= 0:  
    #     # 用i获取当前行的股票代码和转债代码（此时i在有效范围内）
    #     stock = df_input['股票代码'].iloc[i]
    #     bond = df_input['转债代码'].iloc[i]
        
    #     # 后续处理逻辑（示例）
    #     stock_a = stock  # 股票
    #     stock_b = bond   # 转债
    #     print(f"处理第{i}行：股票代码={stock}，转债代码={bond}")  # 可验证是否正确
    #     main_file(stock_a,stock_b,end_date)
    #     # 3. 每次循环让i减1，逐步遍历上一行
    #     i -= 1  