import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from typing import Optional, List, Tuple, Dict, Union

def mom_cumrtn_and_sharpe(
    data_df: pd.DataFrame,
    start: int = 10,
    end: int = 245,
    step: int = 5,
    ex_K: Optional[int] = None,
    rtn_is_log: bool = False,
    rtn_categories: List[str] = ['intra', 'overnight', 'close'],
    rtn_cols: List[str] = ['intra_rtn', 'overnight_rtn', 'close_rtn'],
    feature_types: List[str] = ['cumrtn', 'sharpe'],
    prefix: str = '',
    suffix: str = '',
    verbose: bool = True,
    annualize_sharpe: bool = True
) -> pd.DataFrame:
    """
    生成基于收益率序列的动量特征矩阵
    
    参数:
        data_df: 输入数据框，必须包含rtn_cols中指定的列
        start: 总回看窗口的起始天数（包含ex_K）
        end: 总回看窗口的结束天数（包含ex_K）
        step: 窗口步长
        ex_K: 从当前时点往前剔除的天数（必须小于lookback）
        ...（其他参数不变）...
    """
    # 参数校验
    if len(rtn_categories) != len(rtn_cols):
        raise ValueError("rtn_categories和rtn_cols的长度必须相同")
    if ex_K is not None and ex_K >= start:
        raise ValueError("ex_K必须小于start")
    
    # 转换为对数收益率
    log_returns = {}
    for cat, col in zip(rtn_categories, rtn_cols):
        if col not in data_df.columns:
            raise ValueError(f"数据框中缺少列: {col}")
        log_returns[cat] = data_df[col] if rtn_is_log else np.log(data_df[col] + 1)
    
    X_features_list = []
    
    for rtn_category, rtn_series in log_returns.items():
        for lookback in trange(start, end + step, step, disable=not verbose):
            # 实际计算窗口长度
            calc_window = lookback if ex_K is None else (lookback - ex_K)
            if calc_window <= 0:
                raise ValueError(f"无效窗口: lookback={lookback}, ex_K={ex_K}")
            
            # 特征名称标记
            ex_suffix = f"_ex{ex_K}" if ex_K is not None else ""
            
            # 计算各特征
            if 'cumrtn' in feature_types:
                cum_rtn = rtn_series.rolling(lookback).sum().shift(ex_K) if ex_K is not None else rtn_series.rolling(lookback).sum()
                name = f"{prefix}{rtn_category}cumrtn_LB{lookback}{ex_suffix}{suffix}"
                X_features_list.append(cum_rtn.rename(name))
            
            if 'sharpe' in feature_types:
                def sharpe_fn(x):
                    if x.std() == 0:
                        return 0
                    ratio = x.mean() / x.std()
                    return ratio * (np.sqrt(252) if annualize_sharpe else 1)
                
                sharpe = rtn_series.rolling(lookback).apply(sharpe_fn).shift(ex_K) if ex_K is not None else rtn_series.rolling(lookback).apply(sharpe_fn)
                name = f"{prefix}{rtn_category}sharpe_LB{lookback}{ex_suffix}{suffix}"
                X_features_list.append(sharpe.rename(name))
    
    return pd.concat(X_features_list, axis=1)

def generate_all_term_features(
    data_df: pd.DataFrame,
    short_term_config: Dict = {'start': 10, 'end': 30, 'step': 5, 'ex_K': None},
    medium_term_config: Dict = {'start': 35, 'end': 90, 'step': 5, 'ex_K': 5},
    long_term_config: Dict = {'start': 95, 'end': 245, 'step': 10, 'ex_K': 20},
    rtn_categories: List[str] = ['intra', 'overnight', 'close'],
    rtn_cols: List[str] = ['intra_rtn', 'overnight_rtn', 'close_rtn'],
    feature_types: List[str] = ['cumrtn', 'sharpe'],
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    生成短期、中期和长期的动量特征
    
    参数:
        data_df (pd.DataFrame): 输入数据框，必须包含rtn_cols中指定的列
        short_term_config (Dict): 短期特征配置，包含start, end, step, ex_K
        medium_term_config (Dict): 中期特征配置，包含start, end, step, ex_K
        long_term_config (Dict): 长期特征配置，包含start, end, step, ex_K
        rtn_categories (List[str]): 收益率类别名称
        rtn_cols (List[str]): 收益率列名
        feature_types (List[str]): 特征类型
        verbose (bool): 是否显示进度条
        
    返回:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
            (所有特征, 短期特征, 中期特征, 长期特征)
    """
    # 生成短期特征
    if verbose:
        print("生成短期特征...")
    X_short = mom_cumrtn_and_sharpe(
        data_df=data_df,
        start=short_term_config['start'],
        end=short_term_config['end'],
        step=short_term_config['step'],
        ex_K=short_term_config['ex_K'],
        rtn_categories=rtn_categories,
        rtn_cols=rtn_cols,
        feature_types=feature_types,
        prefix='short_',
        verbose=verbose
    )
    
    # 生成中期特征
    if verbose:
        print("生成中期特征...")
    X_medium = mom_cumrtn_and_sharpe(
        data_df=data_df,
        start=medium_term_config['start'],
        end=medium_term_config['end'],
        step=medium_term_config['step'],
        ex_K=medium_term_config['ex_K'],
        rtn_categories=rtn_categories,
        rtn_cols=rtn_cols,
        feature_types=feature_types,
        prefix='medium_',
        verbose=verbose
    )
    
    # 生成长期特征
    if verbose:
        print("生成长期特征...")
    X_long = mom_cumrtn_and_sharpe(
        data_df=data_df,
        start=long_term_config['start'],
        end=long_term_config['end'],
        step=long_term_config['step'],
        ex_K=long_term_config['ex_K'],
        rtn_categories=rtn_categories,
        rtn_cols=rtn_cols,
        feature_types=feature_types,
        prefix='long_',
        verbose=verbose
    )
    
    # 合并所有特征
    X_all = pd.concat([X_short, X_medium, X_long], axis=1)
    
    return X_all, X_short, X_medium, X_long 