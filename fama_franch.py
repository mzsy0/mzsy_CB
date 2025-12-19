import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from arch import arch_model
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance
import warnings
import traceback
from scipy.optimize import minimize
from cvxopt import matrix, solvers

# 忽略警告
warnings.filterwarnings('ignore', category=UserWarning, module='arch')
solvers.options['show_progress'] = False  # 关闭CVXOPT求解器输出

# 增强型数据加载函数
def load_stock_data(file_path):
    """加载股票数据并计算对数收益率，增加数据验证"""
    try:
        dtype_mapping = {
            '开盘价': 'float32',
            '收盘价': 'float32',
            '最高价': 'float32',
            '最低价': 'float32',
            '成交量': 'float32',
            '成交额': 'float32',
            '换手率': 'float32',
            '交易时间': 'datetime64[ns]'
        }
        data = pd.read_excel(file_path, dtype=dtype_mapping)
        
        # 数据验证
        if data.empty:
            raise ValueError("读取的数据为空")
        
        # 检查关键列是否存在
        required_columns = ['交易时间', '收盘价']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"数据缺少必要的列: {col}")
        
        data.sort_values(by='交易时间', inplace=True)
        
        # 计算对数收益率并处理异常值
        data['stock_return'] = np.log(data['收盘价'] / data['收盘价'].shift(1))
        
        # 处理无穷大或NaN值
        data['stock_return'] = data['stock_return'].replace([np.inf, -np.inf], np.nan)
        data = data.dropna(subset=['stock_return'])
        
        print(f"成功加载股票数据，共{len(data)}条记录")
        return data
    except Exception as e:
        print(f"加载股票数据失败: {e}")
        return pd.DataFrame(columns=['交易时间', '收盘价', 'stock_return'])

def load_market_data(file_path):
    """加载市场数据并计算对数收益率，增加数据验证"""
    try:
        dtype_mapping = {
            '开盘价': 'float32',
            '收盘价': 'float32',
            '交易时间': 'datetime64[ns]'
        }
        data = pd.read_excel(file_path, dtype=dtype_mapping)
        
        # 数据验证
        if data.empty:
            raise ValueError("读取的数据为空")
        
        # 检查关键列是否存在
        required_columns = ['交易时间', '收盘价']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"数据缺少必要的列: {col}")
        
        data.sort_values(by='交易时间', inplace=True)
        data['market_return'] = np.log(data['收盘价'] / data['收盘价'].shift(1))
        
        # 处理无穷大或NaN值
        data['market_return'] = data['market_return'].replace([np.inf, -np.inf], np.nan)
        data = data.dropna(subset=['market_return'])
        
        print(f"成功加载市场数据，共{len(data)}条记录")
        return data
    except Exception as e:
        print(f"加载市场数据失败: {e}")
        return pd.DataFrame(columns=['交易时间', 'market_return'])

def load_factor_data(file_path, factor_type):
    """加载因子数据并计算对数收益率，增加数据验证"""
    try:
        dtype_mapping = {
            '开盘价': 'float32',
            '收盘价': 'float32',
            '交易时间': 'datetime64[ns]'
        }
        data = pd.read_excel(file_path, dtype=dtype_mapping)
        
        # 数据验证
        if data.empty:
            raise ValueError("读取的数据为空")
        
        # 检查关键列是否存在
        required_columns = ['交易时间', '收盘价']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"数据缺少必要的列: {col}")
        
        data.sort_values(by='交易时间', inplace=True)
        data['return'] = np.log(data['收盘价'] / data['收盘价'].shift(1))
        
        # 处理无穷大或NaN值
        data['return'] = data['return'].replace([np.inf, -np.inf], np.nan)
        data = data.dropna(subset=['return'])
        
        data.rename(columns={'return': factor_type}, inplace=True)
        print(f"成功加载{factor_type}因子数据，共{len(data)}条记录")
        return data
    except Exception as e:
        print(f"加载{factor_type}因子数据失败: {e}")
        return pd.DataFrame(columns=['交易时间', factor_type])

def build_smb_factor(small_data, large_data):
    """构建稳健的SMB因子，增加数据验证"""
    try:
        # 数据验证
        if small_data.empty or large_data.empty:
            raise ValueError("SMB因子构建失败：小市值或大市值数据为空")
        
        # 1. 数据对齐
        merged = pd.merge(small_data, large_data, on='交易时间', suffixes=('_small', '_large'), how='inner')
        
        if merged.empty:
            raise ValueError("SMB因子构建失败：小市值和大市值数据无重叠日期")
        
        # 2. 滚动窗口标准化
        merged['small_std'] = merged['return_small'].rolling(window=30, min_periods=10).std()
        merged['large_std'] = merged['return_large'].rolling(window=30, min_periods=10).std()
        
        # 避免除以0
        merged['small_std'] = merged['small_std'].replace(0, 1e-5)
        merged['large_std'] = merged['large_std'].replace(0, 1e-5)
        
        # 3. 波动率调整后的因子
        merged['smb_factor'] = (merged['return_small']/merged['small_std'] - 
                                merged['return_large']/merged['large_std'])
        
        # 4. 缩尾处理
        q_low = merged['smb_factor'].quantile(0.05)
        q_high = merged['smb_factor'].quantile(0.95)
        merged['smb_factor'] = merged['smb_factor'].clip(lower=q_low, upper=q_high)
        
        return merged[['交易时间', 'smb_factor']]
    except Exception as e:
        print(f"构建SMB因子失败: {e}")
        return pd.DataFrame(columns=['交易时间', 'smb_factor'])

def build_hml_factor(high_data, low_data):
    """构建稳健的HML因子，增加数据验证"""
    try:
        # 数据验证
        if high_data.empty or low_data.empty:
            raise ValueError("HML因子构建失败：高BM或低BM数据为空")
        
        # 1. 数据对齐
        merged = pd.merge(high_data, low_data, on='交易时间', suffixes=('_high', '_low'), how='inner')
        
        if merged.empty:
            raise ValueError("HML因子构建失败：高BM和低BM数据无重叠日期")
        
        # 2. 滚动窗口标准化
        merged['high_std'] = merged['return_high'].rolling(window=30, min_periods=10).std()
        merged['low_std'] = merged['return_low'].rolling(window=30, min_periods=10).std()
        
        # 避免除以0
        merged['high_std'] = merged['high_std'].replace(0, 1e-5)
        merged['low_std'] = merged['low_std'].replace(0, 1e-5)
        
        # 3. 波动率调整后的因子
        merged['hml_factor'] = (merged['return_high']/merged['high_std'] - 
                                merged['return_low']/merged['low_std'])
        
        # 4. 缩尾处理
        q_low = merged['hml_factor'].quantile(0.05)
        q_high = merged['hml_factor'].quantile(0.95)
        merged['hml_factor'] = merged['hml_factor'].clip(lower=q_low, upper=q_high)
        
        return merged[['交易时间', 'hml_factor']]
    except Exception as e:
        print(f"构建HML因子失败: {e}")
        return pd.DataFrame(columns=['交易时间', 'hml_factor'])

def calculate_rsi(prices, window=14):
    """计算相对强弱指数(RSI)"""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        
        # 避免除以0
        avg_loss = avg_loss.replace(0, 1e-5)
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except Exception as e:
        print(f"计算RSI失败: {e}")
        return pd.Series(index=prices.index)

def fit_egarch(returns):
    """拟合EGARCH模型，处理波动率聚类"""
    try:
        # 检查数据有效性
        if len(returns) < 30:
            raise ValueError("EGARCH模型需要至少30个数据点")
        
        # 缩放收益率数据以优化拟合
        scaled_returns = returns * 100
        
        # 创建EGARCH模型
        egarch = arch_model(
            scaled_returns, 
            mean='Constant', 
            vol='EGARCH', 
            p=1, 
            q=1,
            dist='ged'
        )
        
        egarch_result = egarch.fit(disp='off')
        
        # 获取条件波动率
        conditional_vol = egarch_result.conditional_volatility / 100
        
        # 安全获取最后一期波动率
        if len(conditional_vol) > 0:
            egarch_daily_vol = conditional_vol[-1]
        else:
            egarch_daily_vol = returns.std()
            print("警告：条件波动率序列为空，使用历史波动率")
        
        egarch_annual_vol = egarch_daily_vol * np.sqrt(252)
        
        return {
            'model': 'EGARCH',
            'params': egarch_result.params,
            'daily_vol': egarch_daily_vol,
            'annual_vol': egarch_annual_vol
        }
    except Exception as e:
        print(f"EGARCH模型拟合失败: {e}，回退到GARCH模型")
        return fit_garch(returns)

def fit_garch(returns):
    """拟合标准GARCH模型"""
    try:
        # 检查数据有效性
        if len(returns) < 30:
            raise ValueError("GARCH模型需要至少30个数据点")
        
        # 缩放收益率数据
        scaled_returns = returns * 100
        
        # 创建GARCH模型
        garch_model = arch_model(
            scaled_returns, 
            mean='Constant', 
            vol='GARCH', 
            p=1, 
            q=1,
            dist='normal'
        )
        
        garch_result = garch_model.fit(disp='off')
        
        # 获取条件波动率
        conditional_vol = garch_result.conditional_volatility / 100
        
        # 安全获取最后一期波动率
        if len(conditional_vol) > 0:
            garch_daily_vol = conditional_vol[-1]
        else:
            garch_daily_vol = returns.std()
            print("警告：条件波动率序列为空，使用历史波动率")
        
        garch_annual_vol = garch_daily_vol * np.sqrt(252)
        
        return {
            'model': 'GARCH',
            'params': garch_result.params,
            'daily_vol': garch_daily_vol,
            'annual_vol': garch_annual_vol
        }
    except Exception as e:
        print(f"GARCH模型拟合失败: {e}，使用EWMA模型替代")
        return fit_ewma(returns)

def fit_ewma(returns):
    """使用指数加权移动平均(EWMA)计算波动率"""
    try:
        # 检查数据有效性
        if len(returns) < 10:
            raise ValueError("EWMA计算需要至少10个数据点")
        
        ewma_vol = returns.ewm(span=30).std().iloc[-1]
        ewma_annual_vol = ewma_vol * np.sqrt(252)
        return {
            'model': 'EWMA',
            'params': None,
            'daily_vol': ewma_vol,
            'annual_vol': ewma_annual_vol
        }
    except Exception as e:
        print(f"EWMA计算失败: {e}，使用简单标准差替代")
        return {
            'model': 'SimpleStd',
            'params': None,
            'daily_vol': returns.std(),
            'annual_vol': returns.std() * np.sqrt(252)
        }

def validate_model(X, y):
    """使用时间序列交叉验证评估模型"""
    try:
        if len(X) < 10:
            return np.nan
        
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # 添加常数项
            X_train_const = sm.add_constant(X_train)
            X_test_const = sm.add_constant(X_test)
            
            model = sm.OLS(y_train, X_train_const).fit()
            pred = model.predict(X_test_const)
            
            # 计算预测与实际的相关性
            valid_mask = ~np.isnan(pred) & ~np.isnan(y_test)
            if np.sum(valid_mask) > 1:  # 至少需要2个点计算相关性
                score = np.corrcoef(pred[valid_mask], y_test[valid_mask])[0, 1]
                scores.append(score)
        
        return np.nanmean(scores) if scores else np.nan
    except Exception as e:
        print(f"模型验证失败: {e}")
        return np.nan

def calculate_shapley_importance(X, y, n_repeats=5):
    """计算因子贡献的Shapley值，使用scikit-learn的线性回归模型"""
    try:
        # 创建并拟合scikit-learn的线性回归模型
        lr_model = LinearRegression()
        lr_model.fit(X, y)
        
        # 使用排列重要性作为Shapley值的近似
        result = permutation_importance(
            lr_model, X, y, n_repeats=n_repeats, random_state=42
        )
        
        # 计算相对重要性
        total_importance = np.sum(result.importances_mean)
        if total_importance == 0:
            return {col: 0 for col in X.columns}
        
        return {col: result.importances_mean[i] / total_importance 
                for i, col in enumerate(X.columns)}
    except Exception as e:
        print(f"Shapley值计算失败: {e}")
        return None

def portfolio_optimization(expected_returns, factor_exposures, factor_cov, residual_risks, risk_aversion=1.0, max_weight=0.1):
    """
    使用二次规划进行组合优化
    :param expected_returns: 各资产的预期收益率向量 (n_assets,)
    :param factor_exposures: 因子暴露矩阵 (n_assets, n_factors)
    :param factor_cov: 因子收益协方差矩阵 (n_factors, n_factors)
    :param residual_risks: 残差风险向量 (n_assets,)
    :param risk_aversion: 风险厌恶系数
    :param max_weight: 单个资产最大权重限制
    :return: 优化后的权重向量
    """
    n_assets = len(expected_returns)
    n_factors = factor_exposures.shape[1]
    
    print(f"\n开始组合优化: {n_assets}个资产, {n_factors}个因子")
    
    # 1. 计算总协方差矩阵
    # 总协方差 = 因子协方差部分 + 残差风险部分
    factor_risk = factor_exposures @ factor_cov @ factor_exposures.T
    residual_risk = np.diag(residual_risks**2)
    total_cov = factor_risk + residual_risk
    
    # 2. 定义二次规划问题
    # 目标函数: min (1/2) * w.T @ (risk_aversion * total_cov) @ w - w.T @ expected_returns
    P = matrix(risk_aversion * total_cov)  # 二次项系数矩阵
    q = matrix(-expected_returns)          # 线性项系数向量
    
    # 3. 约束条件
    # 权重和为1
    A = matrix(np.ones((1, n_assets)))
    b = matrix(1.0)
    
    # 权重非负且不超过上限
    G = matrix(np.vstack((-np.eye(n_assets), np.eye(n_assets))))
    h = matrix(np.hstack((np.zeros(n_assets), max_weight * np.ones(n_assets))))
    
    # 4. 求解二次规划
    solution = solvers.qp(P, q, G, h, A, b)
    
    if solution['status'] != 'optimal':
        print("警告: 二次规划未找到最优解, 使用等权重作为后备方案")
        return np.ones(n_assets) / n_assets
    
    weights = np.array(solution['x']).flatten()
    
    # 5. 后处理 - 确保权重和为1
    weights /= weights.sum()
    
    print("组合优化完成")
    print(f"优化权重: {weights}")
    print(f"权重和: {weights.sum():.6f}")
    print(f"最大权重: {weights.max():.4f}, 最小权重: {weights.min():.4f}")
    
    return weights

def calculate_capm(m, market_return_data, risk_free_rate, way, num, 
                  smb_small_stock_path, smb_large_stock_path, 
                  hml_high_bm_path, hml_low_bm_path):
    """
    计算四因子模型：股票β系数、情绪因子敏感度、SMB因子、HML因子与预期收益率
    """
    # 初始化结果字典
    results = {
        'alpha': np.nan,
        'beta_market': np.nan,
        'beta_sentiment': np.nan,
        'beta_smb': np.nan,
        'beta_hml': np.nan,
        'expected_return': np.nan,
        'residual_std': np.nan,
        'smb_premium': np.nan,
        'hml_premium': np.nan,
        'sentiment_premium': np.nan,
        'daily_volatility': np.nan,
        'annualized_volatility': np.nan,
        'monthly_rate': np.nan,
        'market_expected_return': np.nan,
        'pls_r_squared': np.nan,
        'anova_f': np.nan,
        'anova_p': np.nan,
        'market_contribution': np.nan,
        'sentiment_contribution': np.nan,
        'smb_contribution': np.nan,
        'hml_contribution': np.nan,
        'total_factor_contribution': np.nan,
        'vol_model': 'Unknown',
        'validation_score': np.nan,
        'shapley_importance': None,
        'trading_signal': 0,
        'validation_accuracy': np.nan,
        'inverse_strategy': False,
        'simple_model_name': "Unknown",
        'simple_model_features': [],
        'simple_model_rsquared': np.nan,
        'simple_model_rsquared_adj': np.nan,
        'simple_model_fvalue': np.nan,
        'simple_model_f_pvalue': np.nan,
        'next_prediction': np.nan,
        'optimized_weights': None,
        'portfolio_expected_return': np.nan,
        'portfolio_volatility': np.nan,
        'portfolio_sharpe_ratio': np.nan
    }
    
    try:
        # 使用封装函数加载股票数据
        stock_data = load_stock_data(way)
        
        if stock_data.empty:
            raise ValueError("股票数据为空，无法继续计算")
        
        # 获取最近m天数据（确保有m个交易日）
        stock_data = stock_data.sort_values('交易时间')
        filtered_stock_data = stock_data.tail(m).copy()
        
        if filtered_stock_data.empty:
            raise ValueError(f"筛选后股票数据为空")
        
        print(f"分析日期范围: {filtered_stock_data['交易时间'].min().date()} 至 {filtered_stock_data['交易时间'].max().date()}")
        print(f"股票数据点数: {len(filtered_stock_data)}")
        
        # 计算隔夜收益率和换手率增长额
        filtered_stock_data['overnight_rate'] = (filtered_stock_data['开盘价'] - filtered_stock_data['收盘价'].shift(1)) / filtered_stock_data['收盘价'].shift(1)
        filtered_stock_data['turnover_change'] = (filtered_stock_data['成交量'] - filtered_stock_data['成交量'].shift(1)) / num / 100
        
        # 计算技术指标
        filtered_stock_data['volatility'] = filtered_stock_data['收盘价'].pct_change().rolling(window=5, min_periods=1).std()
        filtered_stock_data['rsi'] = calculate_rsi(filtered_stock_data['收盘价'], window=14)
        
        # 创建特征矩阵 (确保所有特征存在)
        features = filtered_stock_data[['overnight_rate', 'turnover_change', 'volatility', 'rsi']].dropna()
        
        if features.empty:
            raise ValueError("特征矩阵为空，无法继续计算")
        
        # 提取目标变量（股票收益率）
        y_target = filtered_stock_data.loc[features.index, 'stock_return']
        
        if y_target.empty:
            raise ValueError("目标变量为空，无法继续计算")
        
        # 数据标准化（特征和目标）
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        scaled_features = x_scaler.fit_transform(features)
        scaled_y = y_scaler.fit_transform(y_target.values.reshape(-1, 1)).flatten()
        
        # 应用PLS回归分析情绪因子
        pls = PLSRegression(n_components=1)
        pls.fit(scaled_features, scaled_y)
        
        # 获取X得分作为情绪因子
        sentiment_pls = pls.x_scores_[:, 0]
        
        # 将PLS得分作为情绪因子（确保索引对齐）
        filtered_stock_data.loc[features.index, 'sentiment'] = sentiment_pls
        results['pls_r_squared'] = pls.score(scaled_features, scaled_y)
        
        # 加载因子数据
        small_data = load_factor_data(smb_small_stock_path, 'small_return')
        large_data = load_factor_data(smb_large_stock_path, 'large_return')
        high_data = load_factor_data(hml_high_bm_path, 'high_bm_return')
        low_data = load_factor_data(hml_low_bm_path, 'low_bm_return')
        
        # 构建因子
        smb_factor_df = build_smb_factor(
            small_data.rename(columns={'small_return': 'return_small'}),
            large_data.rename(columns={'large_return': 'return_large'})
        )
        
        hml_factor_df = build_hml_factor(
            high_data.rename(columns={'high_bm_return': 'return_high'}),
            low_data.rename(columns={'low_bm_return': 'return_low'})
        )
        
        # 合并因子数据 - 使用内部连接确保日期匹配
        merged_factors = pd.merge(
            smb_factor_df, 
            hml_factor_df, 
            on='交易时间', 
            how='inner'
        )
        
        # 将因子合并到主数据 - 使用内部连接确保日期匹配
        filtered_stock_data = pd.merge(
            filtered_stock_data, 
            merged_factors[['交易时间', 'smb_factor', 'hml_factor']], 
            on='交易时间', 
            how='inner'
        )
        
        # 删除NaN值
        filtered_stock_data = filtered_stock_data.dropna(
            subset=['stock_return', 'sentiment', 'smb_factor', 'hml_factor']
        )
        
        if filtered_stock_data.empty:
            raise ValueError("合并因子后的数据为空，无法继续计算")
        
        # 对齐市场收益率数据 - 使用内部连接确保日期匹配
        merged_data = pd.merge(
            filtered_stock_data[['交易时间', 'stock_return', 'sentiment', 'smb_factor', 'hml_factor']], 
            market_return_data,
            on='交易时间',
            how='inner'
        )
        
        # 清洗合并后的数据
        merged_data = merged_data.dropna(
            subset=['stock_return', 'market_return', 'sentiment', 'smb_factor', 'hml_factor']
        )
        
        if merged_data.empty:
            raise ValueError("合并市场数据后的数据为空，无法继续计算")
        
        print(f"最终用于回归分析的数据点数: {len(merged_data)}")
        
        # 1. 增强数据验证
        def enhanced_data_validation(df, name):
            """执行全面的数据质量检查"""
            print(f"\n执行{name}数据验证...")
            
            # 检查缺失值
            missing = df.isnull().sum()
            if missing.any():
                print(f"警告: {name}中存在缺失值:\n{missing[missing > 0]}")
                df = df.dropna()
            
            # 检查无穷值
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if np.isinf(df[col]).any():
                    print(f"警告: {name}的{col}列存在无穷值")
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                    df = df.dropna(subset=[col])
            
            # 检查零方差特征
            zero_var_cols = [col for col in df.columns if df[col].nunique() == 1]
            if zero_var_cols:
                print(f"警告: {name}中存在零方差特征: {zero_var_cols}")
                df = df.drop(columns=zero_var_cols)
            
            print(f"{name}验证完成，剩余{len(df)}条有效记录")
            return df
        
        merged_data = enhanced_data_validation(merged_data, "合并数据")
        
        if merged_data.empty:
            raise ValueError("数据验证后无有效记录，无法继续计算")
        
        # 2. 简化模型
        def simplified_model(data):
            """尝试更简单的模型配置"""
            print("\n尝试简化模型...")
            
            # 尝试不同模型配置
            models = [
                ('仅市场因子', ['market_return']),
                ('市场+情绪', ['market_return', 'sentiment']),
                ('市场+SMB+HML', ['market_return', 'smb_factor', 'hml_factor']),
                ('全因子', ['market_return', 'sentiment', 'smb_factor', 'hml_factor'])
            ]
            
            best_score = -np.inf
            best_model = None
            best_name = ""
            best_model_features = None
            
            for name, features in models:
                X_sub = data[features]
                X_sub_const = sm.add_constant(X_sub)
                y = data['stock_return']
                
                try:
                    model = sm.OLS(y, X_sub_const).fit()
                    score = validate_model(X_sub, y)
                    
                    if not np.isnan(score) and score > best_score:
                        best_score = score
                        best_model = model
                        best_name = name
                        best_model_features = features
                        
                    print(f"{name}模型验证得分: {score:.4f}")
                except Exception as e:
                    print(f"{name}模型拟合失败: {e}")
            
            if best_model is None:
                print("警告: 所有简化模型均失败")
                return None, "None", []
            
            print(f"选择最佳简化模型: {best_name} (得分: {best_score:.4f})")
            return best_model, best_name, best_model_features
        
        # 使用简化模型
        try:
            best_simple_model, best_name, best_model_features = simplified_model(merged_data)
            
            if best_simple_model is not None:
                # 存储简化模型信息
                results['simple_model_name'] = best_name
                results['simple_model_features'] = best_model_features
                results['simple_model_rsquared'] = best_simple_model.rsquared
                results['simple_model_rsquared_adj'] = best_simple_model.rsquared_adj
                results['simple_model_fvalue'] = best_simple_model.fvalue
                results['simple_model_f_pvalue'] = best_simple_model.f_pvalue
                
                # 预测下一个值
                if len(merged_data) > 0:
                    last_row = merged_data.iloc[-1:][best_model_features].copy()
                    last_row_const = sm.add_constant(last_row, has_constant='add')
                    results['next_prediction'] = best_simple_model.predict(last_row_const)[0]
        except Exception as e:
            print(f"简化模型创建失败: {e}")
            results['simple_model_name'] = "Error"
        
        # 3. 交叉验证
        def enhanced_cross_validation(X, y):
            """增强的交叉验证流程"""
            try:
                if len(X) < 10:
                    return {'accuracy': np.nan}
                
                tscv = TimeSeriesSplit(n_splits=min(5, len(X)//2))
                scores = []
                
                for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    
                    # 添加常数项
                    X_train_const = sm.add_constant(X_train)
                    X_test_const = sm.add_constant(X_test)
                    
                    model = sm.OLS(y_train, X_train_const).fit()
                    predictions = model.predict(X_test_const)
                    
                    # 计算预测方向变化
                    pred_diff = predictions.diff().dropna()
                    actual_diff = y_test.diff().dropna()
                    
                    # 对齐索引
                    common_index = pred_diff.index.intersection(actual_diff.index)
                    if not common_index.empty:
                        pred_diff = pred_diff.loc[common_index]
                        actual_diff = actual_diff.loc[common_index]
                        
                        # 计算方向正确率
                        direction_correct = np.sign(pred_diff) == np.sign(actual_diff)
                        accuracy = direction_correct.mean()
                        
                        if not np.isnan(accuracy):
                            scores.append(accuracy)
                            print(f"Fold {fold+1} 方向预测准确率: {accuracy:.4f}")
                
                return {'accuracy': np.nanmean(scores) if scores else np.nan}
            except Exception as e:
                print(f"交叉验证失败: {e}")
                return {'accuracy': np.nan}
        
        # 执行增强交叉验证
        X_cv = merged_data[['market_return', 'sentiment', 'smb_factor', 'hml_factor']]
        y_cv = merged_data['stock_return']
        cv_results = enhanced_cross_validation(X_cv, y_cv)
        results['validation_accuracy'] = cv_results['accuracy']
        
        # 5. 策略调整配置
        config = {
            'buy_threshold': 0.01,
            'sell_threshold': -0.01,
            'use_inverse_strategy': False
        }
        
        # 根据验证结果调整策略
        if results['validation_accuracy'] < 0.5:
            print("警告：模型预测准确性低于50%，启用反转策略")
            config['use_inverse_strategy'] = True
            results['inverse_strategy'] = True
        else:
            print("使用正常交易策略")
        
        # 交易信号生成
        if not np.isnan(results['next_prediction']):
            if config['use_inverse_strategy']:
                # 反转策略
                if results['next_prediction'] > config['buy_threshold']:
                    results['trading_signal'] = -1  # 卖出信号
                elif results['next_prediction'] < config['sell_threshold']:
                    results['trading_signal'] = 1   # 买入信号
            else:
                # 正常策略
                if results['next_prediction'] > config['buy_threshold']:
                    results['trading_signal'] = 1   # 买入信号
                elif results['next_prediction'] < config['sell_threshold']:
                    results['trading_signal'] = -1  # 卖出信号
        
        # 四因子回归分析
        X = merged_data[['market_return', 'sentiment', 'smb_factor', 'hml_factor']]
        y = merged_data['stock_return']
        
        # 使用时间序列交叉验证评估模型
        results['validation_score'] = validate_model(X, y)
        
        # 使用scikit-learn的线性回归模型替代statsmodels
        lr_model = LinearRegression()
        lr_model.fit(X, y)
        
        # 获取回归系数
        results['alpha'] = lr_model.intercept_
        results['beta_market'] = lr_model.coef_[0]
        results['beta_sentiment'] = lr_model.coef_[1]
        results['beta_smb'] = lr_model.coef_[2]
        results['beta_hml'] = lr_model.coef_[3]
        
        # 计算残差标准差
        residuals = y - lr_model.predict(X)
        results['residual_std'] = residuals.std()
        
        # 计算Shapley重要性
        results['shapley_importance'] = calculate_shapley_importance(X, y)
        
        # 计算因子预期收益率
        if not merged_data.empty:
            def weighted_mean(series):
                n = len(series)
                weights = np.exp(np.linspace(0, 1, n))
                weights /= weights.sum()
                return np.sum(series * weights) if n > 0 else np.nan
            
            # 计算市场预期收益率 (年化)
            market_expected_return = weighted_mean(merged_data['market_return']) * 252
            results['market_expected_return'] = market_expected_return
            
            # 计算因子风险溢价 (年化)
            market_premium = market_expected_return - risk_free_rate
            sentiment_premium = weighted_mean(merged_data['sentiment']) * results['beta_sentiment'] * 252
            smb_premium = weighted_mean(merged_data['smb_factor']) * 252
            hml_premium = weighted_mean(merged_data['hml_factor']) * 252
            
            results['smb_premium'] = smb_premium
            results['hml_premium'] = hml_premium
            results['sentiment_premium'] = sentiment_premium
            
            # 计算股票预期收益率 (年化)
            market_contribution = results['beta_market'] * market_premium
            sentiment_contribution = sentiment_premium
            smb_contribution = results['beta_smb'] * smb_premium
            hml_contribution = results['beta_hml'] * hml_premium
            
            results['market_contribution'] = market_contribution
            results['sentiment_contribution'] = sentiment_contribution
            results['smb_contribution'] = smb_contribution
            results['hml_contribution'] = hml_contribution
            results['total_factor_contribution'] = market_contribution + sentiment_contribution + smb_contribution + hml_contribution
            
            # 计算预期收益率
            results['expected_return'] = (
                risk_free_rate + 
                market_contribution + 
                sentiment_contribution + 
                smb_contribution + 
                hml_contribution
            )
            
            # 计算月度收益率
            results['monthly_rate'] = (1 + results['expected_return']/12) ** 12
        
        # 计算波动率
        returns = merged_data['stock_return'].dropna()
        
        if len(returns) > 10:  # 确保有足够的数据点计算波动率
            # 历史波动率计算
            daily_vol = returns.std()
            annual_vol = daily_vol * np.sqrt(252)
            
            # 使用EGARCH模型拟合波动率
            vol_result = fit_egarch(returns)
            
            # 使用GARCH/EWMA波动率作为最终波动率
            results['annualized_volatility'] = vol_result['annual_vol']
            results['daily_volatility'] = vol_result['daily_vol']
            results['vol_model'] = vol_result['model']
        
        # ==============================================================
        # 组合优化部分 - 修复了因子暴露矩阵构建的语法错误
        # ==============================================================
        try:
            print("\n开始组合优化准备...")
            
            # 1. 准备多资产数据 (假设我们处理多只股票)
            # 在实际应用中，这里应该循环处理多只股票
            # 为简化演示，我们假设只有当前这一只股票
            n_assets = 1  # 实际应用中应大于1
            
            # 2. 收集因子暴露 - 修复了括号不匹配的语法错误
            factor_exposures = np.array([
                [results['beta_market'], 
                 results['beta_sentiment'], 
                 results['beta_smb'], 
                 results['beta_hml']]
            ])  # 形状应为 (n_assets, n_factors)
            
            # 3. 收集预期收益率
            expected_returns = np.array([results['expected_return']])
            
            # 4. 计算残差风险 (年化标准差)
            residual_risks = np.array([results['residual_std'] * np.sqrt(252)])
            
            # 5. 计算因子收益协方差矩阵
            # 因子收益: 市场收益、情绪因子、SMB、HML
            factor_returns = merged_data[['market_return', 'sentiment', 'smb_factor', 'hml_factor']]
            
            # 计算因子收益协方差矩阵 (年化)
            factor_cov = factor_returns.cov() * 252
            
            print("\n因子协方差矩阵:")
            print(factor_cov)
            
            # 6. 执行组合优化
            if n_assets > 1:
                # 实际多资产情况
                weights = portfolio_optimization(
                    expected_returns,
                    factor_exposures,
                    factor_cov.values,
                    residual_risks,
                    risk_aversion=1.5,
                    max_weight=0.2
                )
                
                # 计算组合指标
                portfolio_return = weights @ expected_returns
                portfolio_vol = np.sqrt(weights @ (factor_exposures @ factor_cov.values @ factor_exposures.T + np.diag(residual_risks**2)) @ weights)
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol
                
                results['optimized_weights'] = weights
                results['portfolio_expected_return'] = portfolio_return
                results['portfolio_volatility'] = portfolio_vol
                results['portfolio_sharpe_ratio'] = sharpe_ratio
            else:
                # 单资产情况 - 无法进行组合优化
                print("警告: 只有单一资产，无法进行组合优化")
                results['optimized_weights'] = [1.0]
                results['portfolio_expected_return'] = results['expected_return']
                results['portfolio_volatility'] = results['annualized_volatility']
                results['portfolio_sharpe_ratio'] = (results['expected_return'] - risk_free_rate) / results['annualized_volatility'] if results['annualized_volatility'] > 0 else np.nan
        
        except Exception as e:
            print(f"组合优化失败: {e}")
            traceback.print_exc()
            results['optimized_weights'] = None
            results['portfolio_expected_return'] = np.nan
            results['portfolio_volatility'] = np.nan
            results['portfolio_sharpe_ratio'] = np.nan
        
        return results
    
    except Exception as e:
        print(f"\n模型计算失败: {str(e)}")
        traceback.print_exc()
        return results

def visualize_results(results, stock_data, market_data_full, start_date, end_date):
    """可视化分析结果"""
    # 设置中文字体（选择系统支持的字体）
    # 在 visualize_results 函数开头修改字体设置
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'STSong']  # 微软雅黑优先
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.rcParams['font.size'] = 8  # 固定小号字体
    
    try:
        # 筛选在分析时间范围内的股票数据
        stock_data_in_range = stock_data[
            (stock_data['交易时间'] >= start_date) & 
            (stock_data['交易时间'] <= end_date)
        ]
        
        # 筛选在分析时间范围内的市场数据
        market_data_in_range = market_data_full[
            (market_data_full['交易时间'] >= start_date) & 
            (market_data_full['交易时间'] <= end_date)
        ]
        
        if stock_data_in_range.empty or market_data_in_range.empty:
            print("无法可视化：分析期间内的数据为空")
            return
        
        plt.figure(figsize=(18, 16))
        plt.suptitle('四因子模型分析结果', fontsize=20, fontweight='bold')
        
        # 1. 因子暴露分析
        plt.subplot(3, 2, 1)
        factors = ['市场因子β', '情绪因子β', 'SMB因子β', 'HML因子β']
        betas = [
            results.get('beta_market', 0),
            results.get('beta_sentiment', 0),
            results.get('beta_smb', 0),
            results.get('beta_hml', 0)
        ]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bars = plt.bar(factors, betas, color=colors)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.title('因子暴露分析', fontsize=14)
        plt.ylabel('β系数', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 2. 因子贡献分析
        plt.subplot(3, 2, 2)
        contributions = [
            results.get('market_contribution', 0) * 100,
            results.get('sentiment_contribution', 0) * 100,
            results.get('smb_contribution', 0) * 100,
            results.get('hml_contribution', 0) * 100
        ]
        
        # 修正负值问题：将负贡献设为0
        contributions = [max(c, 0) for c in contributions]
        
        # 过滤掉接近0的值
        labels = []
        sizes = []
        for factor, size in zip(factors, contributions):
            if size > 0.01:  # 只显示有显著贡献的因子
                labels.append(factor)
                sizes.append(size)
        
        if not sizes:
            sizes = [100]
            labels = ['无显著因子贡献']
        
        # 如果所有值都是0，则使用均匀分布避免错误
        if sum(sizes) == 0:
            sizes = [100]
            labels = ['无显著因子贡献']
        
        explode = [0.05 if size > max(sizes)*0.5 else 0 for size in sizes]
        plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90, colors=colors[:len(sizes)])
        plt.title('预期收益率因子贡献', fontsize=14)
        plt.axis('equal')

        # 3. 波动率分析
        plt.subplot(3, 2, 3)
        if not stock_data_in_range.empty:
        # 计算历史波动率
            stock_data_in_range['volatility'] = stock_data_in_range['stock_return'].rolling(window=21).std() * np.sqrt(252)
            
            # 绘制波动率曲线
            plt.plot(stock_data_in_range['交易时间'], stock_data_in_range['volatility'], 
                    label='历史波动率', color='#1f77b4', alpha=0.7)
            
            # 标记当前波动率 - 使用文字描述
            current_vol = results.get('annualized_volatility', 0)
            if current_vol > 0:
                plt.axhline(y=current_vol, color='r', linestyle='--', 
                            label=f'当前波动率: {current_vol:.2%}')
            
            # 简化标签避免数学符号
            plt.title('年化波动率', fontsize=14)
            plt.ylabel('波动率', fontsize=12)
            plt.xlabel('日期', fontsize=12)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
        
        # 4. 预期收益率分解
        plt.subplot(3, 2, 4)
        components = [
            ('无风险利率', results.get('risk_free_rate', 0) * 100),
            ('市场因子', results.get('market_contribution', 0) * 100),
            ('情绪因子', results.get('sentiment_contribution', 0) * 100),
            ('SMB因子', results.get('smb_contribution', 0) * 100),
            ('HML因子', results.get('hml_contribution', 0) * 100)
        ]
        
        # 只显示非零组件
        components = [(name, value) for name, value in components if abs(value) > 0.01]
        
        if not components:
            components = [('无数据', 100)]
        
        names = [c[0] for c in components]
        values = [c[1] for c in components]
        
        y_pos = np.arange(len(names))
        plt.barh(y_pos, values, color=['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd'][:len(names)])
        plt.yticks(y_pos, names)
        plt.xlabel('贡献百分比 (%)')
        plt.title('预期收益率分解', fontsize=14)
        
        # 添加数值标签
        for i, v in enumerate(values):
            plt.text(v + 0.5 if v >= 0 else v - 2, i, f'{v:.2f}%', color='black', va='center')
        
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # 5. 交易信号分析
        plt.subplot(3, 2, 5)
        if not stock_data_in_range.empty and not market_data_in_range.empty:
            # 合并股票和市场数据
            merged = pd.merge(stock_data_in_range, market_data_in_range, on='交易时间', how='inner', suffixes=('_stock', '_market'))
            
            # 计算累计收益率
            merged['stock_cum_return'] = (1 + merged['stock_return']).cumprod()
            if 'market_return' in merged:
                merged['market_cum_return'] = (1 + merged['market_return']).cumprod()
            
            # 绘制累计收益率
            plt.plot(merged['交易时间'], merged['stock_cum_return'], 
                    label='股票累计收益', linewidth=2, color='#1f77b4')
            
            if 'market_cum_return' in merged:
                plt.plot(merged['交易时间'], merged['market_cum_return'], 
                        label='市场累计收益', linewidth=2, color='#ff7f0e', alpha=0.7)
            
            # 标记交易信号
            signal = results.get('trading_signal', 0)
            last_date = merged['交易时间'].iloc[-1]
            last_price = merged['stock_cum_return'].iloc[-1]
            
            if signal == 1:  # 买入信号
                plt.plot(last_date, last_price, '^', markersize=12, color='g', 
                        label='买入信号', alpha=0.8)
                plt.annotate('买入', xy=(last_date, last_price),
                            xytext=(last_date - pd.Timedelta(days=2), last_price * 1.05),
                            arrowprops=dict(facecolor='green', shrink=0.05),
                            fontsize=12, color='green')
            elif signal == -1:  # 卖出信号
                plt.plot(last_date, last_price, 'v', markersize=12, color='r', 
                        label='卖出信号', alpha=0.8)
                plt.annotate('卖出', xy=(last_date, last_price),
                            xytext=(last_date - pd.Timedelta(days=2), last_price * 0.95),
                            arrowprops=dict(facecolor='red', shrink=0.05),
                            fontsize=12, color='red')
            
            plt.title('累计收益率与交易信号', fontsize=14)
            plt.ylabel('累计收益', fontsize=12)
            plt.xlabel('日期', fontsize=12)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
        
        # 6. 模型验证指标
        plt.subplot(3, 2, 6)
        metrics = {
            '模型验证得分': results.get('validation_score', 0),
            '方向预测准确率': results.get('validation_accuracy', 0),
            'PLS R²': results.get('pls_r_squared', 0),
            '简化模型R²': results.get('simple_model_rsquared', 0)
        }
        
        # 过滤无效值
        metrics = {k: v for k, v in metrics.items() if not np.isnan(v)}
        
        if metrics:
            labels = list(metrics.keys())
            values = list(metrics.values())
            
            y_pos = np.arange(len(labels))
            bars = plt.barh(y_pos, values, color='#9467bd')
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                width = bar.get_width()
                plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{value:.4f}', va='center', fontsize=10)
            
            plt.yticks(y_pos, labels)
            plt.title('模型验证指标', fontsize=14)
            plt.xlim(0, 1)
            plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # 添加整体说明
        plt.figtext(0.5, 0.01, 
                    f"股票预期收益率: {results.get('expected_return', 0):.2%} | "
                    f"年化波动率: {results.get('annualized_volatility', 0):.2%} | "
                    f"交易信号: {'买入' if results.get('trading_signal', 0) == 1 else '卖出' if results.get('trading_signal', 0) == -1 else '持有'}",
                    ha="center", fontsize=14, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('四因子模型分析结果.png', dpi=300, bbox_inches='tight')
        # plt.show()
    
    except Exception as e:
        print(f"可视化失败: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # 定义文件路径
    stock_way = r'E:\Python\pythoncode\Data\K线导出_601138_日线数据.xlsx'
    market_way = r'E:\Python\pythoncode\Data\K线导出_000985_日线数据.xlsx'
    
    # 使用封装函数加载数据
    stock_data = load_stock_data(stock_way)
    market_data_full = load_market_data(market_way)
    
    if stock_data.empty or market_data_full.empty:
        print("无法继续：股票数据或市场数据为空")
        exit(1)
    
    # 安全获取输入
    while True:
        try:
            m = int(input("请输入观察天数(整数): "))
            if m <= 0:
                raise ValueError("观察天数必须大于0")
            break
        except ValueError as e:
            print(f"输入错误: {e}, 请重新输入")
    
    end_date = stock_data['交易时间'].max()
    start_date = end_date - pd.Timedelta(days=m)
    
    # 筛选市场数据日期范围
    market_data = market_data_full[
        (market_data_full['交易时间'] >= start_date) & 
        (market_data_full['交易时间'] <= end_date)
    ].copy()
    
    # 添加日期范围检查
    print(f"分析日期范围: {start_date.date()} 至 {end_date.date()}")
    
    # 修复这里的括号匹配问题
    condition = (stock_data['交易时间'] >= start_date) & (stock_data['交易时间'] <= end_date)
    count = len(stock_data[condition])
    print(f"股票数据点数: {count}")
    
    print(f"市场数据点数: {len(market_data)}")
    
    # 安全获取无风险收益率
    while True:
        try:
            risk_free = float(input("请输入无风险年化收益率(小数形式): "))
            if risk_free < 0 or risk_free > 1:
                print("警告: 无风险收益率应在0到1之间")
            break
        except ValueError:
            print("输入错误，请输入小数形式")
    
    # 安全获取发行总股数
    while True:
        try:
            total_shares_in_hundred_millions = float(input("请输入发行总股数(亿): "))
            if total_shares_in_hundred_millions <= 0:
                print("警告: 发行总股数必须大于0")
            else:
                num = int(total_shares_in_hundred_millions * 100000000)  # 将亿转换为实际股数
                break
        except ValueError:
            print("输入错误，请输入数字形式(如10.5)")
    
    # 定义因子文件路径
    smb_small_path = r'E:\Python\pythoncode\Data\K线导出_000852_日线数据.xlsx'
    smb_large_path = r'E:\Python\pythoncode\Data\K线导出_000016_日线数据.xlsx'
    hml_high_path = r'E:\Python\pythoncode\Data\K线导出_KWEB_日线数据.xlsx'
    hml_low_path = r'E:\Python\pythoncode\Data\K线导出_000001_日线数据.xlsx'
    
    # 执行四因子模型计算
    results = calculate_capm(
        m, market_data, risk_free, stock_way, num, 
        smb_small_path, smb_large_path, hml_high_path, hml_low_path
    )
    
    # 将无风险利率添加到结果中，用于可视化
    results['risk_free_rate'] = risk_free
    
    # 可视化结果 - 传入整个数据集和时间范围
    visualize_results(results, stock_data, market_data_full, start_date, end_date)
    
    # 输出四因子模型结果
    print("\n" + "="*50)
    print("四因子模型计算结果")
    print("="*50)
    
    # 输出最佳简化模型结果
    print("\n" + "="*50)
    print("最佳简化模型计算结果")
    print("="*50)
    print(f"模型名称: {results.get('simple_model_name', '未知')}")
    
    # 输出模型包含的因子
    if 'simple_model_features' in results:
        print("\n模型包含的因子:")
        for feature in results['simple_model_features']:
            print(f"- {feature}")
    
    # 输出模型性能指标
    if 'simple_model_rsquared' in results:
        print(f"\n模型R²: {results['simple_model_rsquared']:.4f}")
        print(f"调整R²: {results.get('simple_model_rsquared_adj', 'N/A'):.4f}")
        print(f"F统计量: {results.get('simple_model_fvalue', 'N/A'):.4f}")
        print(f"F统计量p值: {results.get('simple_model_f_pvalue', 'N/A'):.4g}")
    
    # 输出交易信号
    print(f"\n方向预测准确率: {results.get('validation_accuracy', 'N/A'):.4f}")
    print(f"下一个预测值: {results.get('next_prediction', 'N/A'):.6f}")
    
    signal = results.get('trading_signal', 0)
    signal_str = '买入' if signal == 1 else '卖出' if signal == -1 else '持有'
    print(f"交易信号: {signal_str}")
    
    print(f"使用反转策略: {'是' if results.get('inverse_strategy', False) else '否'}")
    
    # 输出四因子模型计算结果
    print("\n" + "="*50)
    print("四因子模型计算结果")
    print("="*50)
    print("证券名称:", stock_data['证券名称'].iloc[0])
    print(f"Alpha系数: {results.get('alpha', 'N/A'):.6f}")
    print(f"市场因子β系数: {results.get('beta_market', 'N/A'):.4f}")
    print(f"情绪因子敏感度: {results.get('beta_sentiment', 'N/A'):.4f}")
    print(f"SMB因子β系数: {results.get('beta_smb', 'N/A'):.4f}")
    print(f"HML因子β系数: {results.get('beta_hml', 'N/A'):.4f}")
    print(f"股票预期年化收益率: {results.get('expected_return', 'N/A'):.4%}")
    print(f"残差标准差: {results.get('residual_std', 'N/A'):.6f}")
    print(f"SMB因子风险溢价: {results.get('smb_premium', 'N/A'):.4%}")
    print(f"HML因子风险溢价: {results.get('hml_premium', 'N/A'):.4%}")
    print(f"情绪因子风险溢价: {results.get('sentiment_premium', 'N/A'):.4%}")
    print(f"股票月度预期收益率: {results.get('monthly_rate', 'N/A'):.4f}%")
    print(f"历史日波动率: {results.get('daily_volatility', 'N/A'):.6f}")
    print(f"历史年化波动率: {results.get('annualized_volatility', 'N/A'):.6f}")
    print(f"使用的波动率模型: {results.get('vol_model', '未知')}")
    print(f"市场预期年化收益率: {results.get('market_expected_return', 'N/A'):.4%}")
    print(f"PLS回归R平方值: {results.get('pls_r_squared', 'N/A'):.4f}")
    print(f"回归模型F统计量: {results.get('anova_f', 'N/A'):.4f}")
    print(f"回归模型p值: {results.get('anova_p', 'N/A'):.4g}")
    print(f"原始模型验证得分: {results.get('validation_score', 'N/A'):.4f}")
    print(f"方向预测准确率: {results.get('validation_accuracy', 'N/A'):.4f}")
    print(f"使用反转策略: {'是' if results.get('inverse_strategy', False) else '否'}")
    print(f"交易信号: {'买入' if signal == 1 else '卖出' if signal == -1 else '持有'}")
    
    # 输出因子贡献分析
    print("\n" + "="*50)
    print("因子贡献分析")
    print("="*50)
    print(f"市场因子贡献: {results.get('market_contribution', 'N/A'):.4%}")
    print(f"情绪因子贡献: {results.get('sentiment_contribution', 'N/A'):.4%}")
    print(f"SMB因子贡献: {results.get('smb_contribution', 'N/A'):.4%}")
    print(f"HML因子贡献: {results.get('hml_contribution', 'N/A'):.4%}")
    print(f"总因子贡献: {results.get('total_factor_contribution', 'N/A'):.4%}")
    
    # 输出Shapley重要性分析
    if 'shapley_importance' in results and results['shapley_importance']:
        print("\n" + "="*50)
        print("因子重要性分析 (Shapley值)")
        print("="*50)
        for factor, importance in results['shapley_importance'].items():
            print(f"{factor}: {importance:.4f}")
    
    # 输出组合优化结果
    print("\n" + "="*50)
    print("组合优化结果")
    print("="*50)
    
    if results['optimized_weights'] is not None:
        print(f"优化权重: {results['optimized_weights']}")
        print(f"组合预期年化收益率: {results.get('portfolio_expected_return', 'N/A'):.4%}")
        print(f"组合预期年化波动率: {results.get('portfolio_volatility', 'N/A'):.4%}")
        print(f"组合夏普比率: {results.get('portfolio_sharpe_ratio', 'N/A'):.4f}")
    else:
        print("组合优化未执行或无结果")