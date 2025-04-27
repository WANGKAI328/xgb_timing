import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from tqdm import tqdm, trange
import os
from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize
from sklearn.model_selection import GridSearchCV
from sklearn.covariance import LedoitWolf
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
import scipy.stats as ss
from sklearn.metrics import mutual_info_score

####################################################################
# 1. Denoising and Detoning Covariance Matrix
####################################################################

def cov2corr(cov):
    """
    将协方差矩阵转换为相关矩阵
    将协方差矩阵转换为相关系数矩阵
    
    参数:
        cov: 输入的协方差矩阵
    
    返回:
        相关系数矩阵
    """
    std = np.sqrt(np.diag(cov))  # 计算标准差 np.diag返回对角线元素
    corr = cov / np.outer(std, std)  # 归一化协方差矩阵
    corr[corr < -1], corr[corr > 1] = -1, 1  # 修正数值误差
    return corr

def corr2cov(corr, std):
    """
    将相关系数矩阵转化为协方差矩阵
    
    参数:
        corr: 相关系数矩阵
        std: 标准差向量
        
    返回:
        协方差矩阵
    """
    cov = corr * np.outer(std, std)
    return cov

def mpPDF(var, q, pts):
    """
    计算Marčenko-Pastur分布的概率密度函数(pdf)
    
    Marcenko-Pastur分布描述的是一个随机矩阵特征值的分布: 
    f[λ] = (T/N) * sqrt((λ_plus - λ)(λ - λ_minus)) / (2πλσ^2) , if λ_minus ≤ λ ≤ λ_plus
    
    参数:
        var: 方差σ^2
        q: 矩阵维度比 T/N
        pts: 生成pdf的点数
        
    返回:
        包含pdf值的Series
    """
    if type(var) not in [int, float]:
        var = var[0]
    # 计算特征值的上下边界
    eMin = var * (1 - (1. / q) ** 0.5) ** 2  # 特征值的下界
    eMax = var * (1 + (1. / q) ** 0.5) ** 2  # 特征值的上界
    # 在上下边界之间均匀生成pts个点
    eVal = np.linspace(eMin, eMax, pts)
    # 计算Marčenko-Pastur分布的pdf (q = T / N )
    pdf = q / (2 * np.pi * var * eVal) * ((eMax - eVal) * (eVal - eMin)) ** 0.5
    # 将pdf值存储为pandas Series，方便后续处理
    pdf = pd.Series(pdf, index=eVal)
    return pdf

def getPCA(matrix):
    """
    对Hermitian矩阵（如相关矩阵）进行特征分解
    
    参数:
        matrix: 输入的Hermitian矩阵(如相关矩阵)
        
    返回:
        特征值和特征向量
    """
    # 计算特征值和特征向量
    eVal, eVec = np.linalg.eigh(matrix)
    # 对特征值进行从大到小排序，并调整特征向量的顺序
    indices = eVal.argsort()[::-1]  # 获取从大到小排序的顺序
    eVal, eVec = eVal[indices], eVec[:, indices]
    # 将特征值转换为对角矩阵形式
    eVal = np.diagflat(eVal)
    return eVal, eVec

def find_optimal_bandwidth(obs, kernel='gaussian', cv=5, bandwidth_range=None):
    """
    通过交叉验证找到最优带宽
    
    参数:
        obs: 观测值数组
        kernel: 核函数类型（默认为高斯核）
        cv: 交叉验证的折数
        bandwidth_range: 带宽搜索范围(默认为 [0.01, 1.0])
        
    返回:
        最优带宽
    """
    if bandwidth_range is None:
        bandwidth_range = np.linspace(start=0.01, stop=1, num=19)  # 默认带宽搜索范围

    # 确保观测值是二维数组形式
    if len(obs.shape) == 1:
        obs = obs.reshape(-1, 1)

    # 使用 GridSearchCV 搜索最优带宽, GridSearchCV对于KernelDensity的评价是log-likelyhood
    grid = GridSearchCV(KernelDensity(kernel=kernel),
                      {'bandwidth': bandwidth_range},
                      cv=cv)
    grid.fit(obs)

    # 返回最优带宽
    return grid.best_params_['bandwidth']

def fitKDE(obs, bWidth=None, kernel='gaussian', x=None):
    """
    使用核密度估计拟合观测值的分布
    
    参数:
        obs: 观测值数组
        bWidth: 核密度估计的带宽（如果为 None，则通过交叉验证选择最优带宽）
        kernel: 核函数类型（默认为高斯核）
        x: 拟合的x值范围（默认为观测值的唯一值）
        
    返回:
        拟合后的概率密度函数
    """
    # 确保观测值是二维数组形式
    if len(obs.shape) == 1:
        obs = obs.reshape(-1, 1)
    # 如果未指定带宽，则通过交叉验证选择最优带宽
    if bWidth is None:
        bWidth = find_optimal_bandwidth(obs, kernel=kernel)
    # 初始化核密度估计器
    kde = KernelDensity(kernel=kernel, bandwidth=bWidth).fit(obs)
    # 如果未指定x值范围，则使用观测值的唯一值
    if x is None:
        x = np.unique(obs).reshape(-1, 1)
    # 确保x值是二维数组形式
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    # 计算核密度估计的对数概率密度
    logProb = kde.score_samples(x)  # 返回对数概率密度
    # 将对数概率密度转换为概率密度
    pdf = pd.Series(np.exp(logProb), index=x.flatten())
    return pdf

def errPDFs(var, eVal, q, bWidth, pts=1000):
    """
    计算理论Marcenko-Pastur分布与经验分布之间的误差
    
    参数:
        var: 方差σ^2
        eVal: 特征值数组
        q: 矩阵维度比 T/N
        bWidth: 核密度估计的带宽
        pts: 生成pdf的点数
        
    返回:
        平方误差和
    """
    pdf0 = mpPDF(var, q, pts)  # 理论pdf
    pdf1 = fitKDE(eVal, bWidth, x=pdf0.index.values)  # 经验pdf
    sse = np.sum((pdf1 - pdf0) ** 2)  # 计算平方误差
    return sse

def findMaxEval(eVal, q, bWidth):
    """
    通过拟合Marčenko-Pastur分布找到最大随机特征值
    
    参数:
        eVal: 特征值数组
        q: 矩阵维度比 T/N
        bWidth: 核密度估计的带宽
        
    返回:
        最大随机特征值和调整后的方差σ^2
    """
    # 尝试找到最合适的Variance
    out = minimize(lambda var: errPDFs(var, eVal, q, bWidth), x0=0.5, bounds=((1E-5, 1 - 1E-5),))
    if out['success']:
        var = out['x'][0]  # 调整后的方差σ^2
    else:
        var = 1  # 如果优化失败，使用默认值
    eMax = var * (1 + (1. / q) ** 0.5) ** 2  # 计算最大随机特征值 eigenMax
    return eMax, var

def denoisedCorr_Mean(eVal, eVec, nFacts):
    """
    使用常数残差特征值法去除噪声
    
    参数:
        eVal: 特征值矩阵（对角矩阵）
        eVec: 特征向量矩阵
        nFacts: 信号特征值的数量
        
    返回:
        去噪后的相关矩阵
    """
    eVal_ = np.diag(eVal).copy()  # 提取特征值（对角线元素）
    eVal_[nFacts:] = eVal_[nFacts:].sum() / float(eVal_.shape[0] - nFacts)  # 将噪声特征值替换为平均值
    eVal_ = np.diag(eVal_)  # 将特征值重新转换为对角矩阵
    corr1 = np.dot(eVec, eVal_).dot(eVec.T)  # 重构相关矩阵
    corr1 = cov2corr(corr1)  # 将重构后的矩阵转换为相关矩阵
    return corr1

def denoisedCorr_Shrink(eVal, eVec, nFacts, alpha=0.8):
    """
    使用目标收缩法去除噪声
    
    参数:
        eVal: 特征值矩阵（对角矩阵）
        eVec: 特征向量矩阵
        nFacts: 信号特征值的数量
        alpha: 收缩参数（0表示完全收缩，1表示不收缩）
        
    返回:
        去噪后的相关矩阵
    """
    eValL, eVecL = eVal[:nFacts, :nFacts], eVec[:, :nFacts]  # 信号部分
    eValR, eVecR = eVal[nFacts:, nFacts:], eVec[:, nFacts:]  # 噪声部分
    corr0 = np.dot(eVecL, eValL).dot(eVecL.T)  # 信号部分的相关矩阵
    corr1 = np.dot(eVecR, eValR).dot(eVecR.T)  # 噪声部分的相关矩阵
    corr2 = corr0 + alpha * corr1 + (1 - alpha) * np.diag(np.diag(corr1))  # 结合信号和噪声部分
    return corr2

def get_denoised_covariance_matrix(df, denoise_method='mean', estimate_raw_cov_method='sample', bWidth=None, denoise_shrink_alpha=0.8, return_cov=True):
    """
    获取去噪后的协方差矩阵
    
    参数:
        df: 输入的DataFrame，每一列是一个变量，每一行是一个观测值
        denoise_method: 去噪方法，'mean' 或 'shrink'
        estimate_raw_cov_method: 估计原始协方差矩阵的方法，'sample' 或 'shrink'
        bWidth: 核密度估计的带宽（如果为 None，则通过交叉验证选择最优带宽）
        denoise_shrink_alpha: 收缩参数，仅在 denoise_method='shrink' 时使用
        return_cov: 是否返回协方差矩阵（True）或相关矩阵（False）
        
    返回:
        去噪后的协方差矩阵或相关矩阵
    """
    nFact, nCols = df.shape[0], df.shape[1] # nFact是行数(Time), nCols是列数(Variable)
    q = nFact / nCols  # 计算矩阵维度比 T/N
    df_columns = df.columns

    # 估计原始协方差矩阵
    if estimate_raw_cov_method == 'sample':
        raw_cov = df.cov()
    elif estimate_raw_cov_method == 'shrink':
        raw_cov = LedoitWolf().fit(df).covariance_
    else:
        raise ValueError("estimate_raw_cov_method 必须是 'sample' 或 'shrink'")

    '''将covariance matrix转化为correlation matrix, 消除各个方差的影响'''
    raw_corr = cov2corr(raw_cov)  # 将协方差矩阵转换为相关矩阵
    eVal_raw, eVec_raw = getPCA(raw_corr)  # 获取原始相关矩阵的特征值和特征向量

    # 找到最大随机特征值
    eMax_Random, var = findMaxEval(np.diag(eVal_raw), q, bWidth=bWidth)
    non_random_eVal_num = np.sum(np.diag(eVal_raw) > eMax_Random)  # 计算非随机特征值的数量
    print("Non-Random Eigen Value num: ", non_random_eVal_num)
    # 去噪相关矩阵
    if denoise_method == 'mean':
        corr_denoised = denoisedCorr_Mean(eVal_raw, eVec_raw, non_random_eVal_num)
    elif denoise_method == 'shrink':
        corr_denoised = denoisedCorr_Shrink(eVal_raw, eVec_raw, non_random_eVal_num, alpha=denoise_shrink_alpha)
    else:
        raise ValueError("denoise_method 必须是 'mean' 或 'shrink'")

    # 将去噪后的相关矩阵转换为协方差矩阵
    if return_cov:
        cov_denoised = corr2cov(corr_denoised, np.diag(raw_cov) ** 0.5)
        return pd.DataFrame(index=df_columns, columns=df_columns, data=cov_denoised)
    else:
        return pd.DataFrame(index=df_columns, columns=df_columns, data=corr_denoised)

def get_detoned_covariance_matrix(denoised_covariance_matrix, market_component_num=1):
    """
    去除市场成分(detoning), 虽然结果出来是singular的,但是clustering等算法不需要invertible
    
    参数:
        denoised_correlation_matrix: 降噪后的相关矩阵
        market_component_num: 市场成分的数量（默认去除最大的1个特征值）
        
    返回:
        去市场成分后的协方差矩阵
    """
    denoised_correlation_matrix = cov2corr(denoised_covariance_matrix)
    # 获取降噪后相关矩阵的特征值和特征向量
    eVal, eVec = getPCA(denoised_correlation_matrix)
    # 提取市场成分的特征值和特征向量
    eVal_market = eVal[:market_component_num, :market_component_num]  # 市场成分的特征值
    eVec_market = eVec[:, :market_component_num]  # 市场成分的特征向量
    # 计算市场成分矩阵
    market_component_matrix = eVec_market @ eVal_market @ eVec_market.T
    # 去除市场成分
    detoned_correlation_matrix = denoised_correlation_matrix - market_component_matrix
    # 标准化相关矩阵（确保对角线元素为1）
    diag_sqrt = np.diag(1 / np.sqrt(np.diag(detoned_correlation_matrix)))
    detoned_correlation_matrix = diag_sqrt @ detoned_correlation_matrix @ diag_sqrt
    # 转化回covariance matrix
    std = np.diag(denoised_covariance_matrix) ** 0.5
    detoned_covariance_matrix = corr2cov(detoned_correlation_matrix, std)
    # return detoned_covariance_matrix    
    return pd.DataFrame(index=denoised_covariance_matrix.index, columns=denoised_covariance_matrix.columns, data=detoned_covariance_matrix.to_numpy())

####################################################################
# 示例使用代码
####################################################################

if __name__ == "__main__":
    # 读取数据
    Data_Saving_Folder = os.path.abspath("..\\Data")  # 数据储存路径
    code_desc_df = pd.read_csv(os.path.join(Data_Saving_Folder, 'code_desc_df.csv'))  # 行业代码描述表
    sw_code_rename_dict = {row.Code: row.SEC_NAME for row_idx, row in code_desc_df.iterrows()}
    sw_l1_folder = os.path.join(Data_Saving_Folder, '申万一级')
    sw_l1_close = pd.read_csv(os.path.join(sw_l1_folder, '收盘价.csv'), parse_dates=['Date'], index_col=0)
    sw_l1_close = sw_l1_close.rename(columns=sw_code_rename_dict)  # 重命名
    sw_l1_rtn = sw_l1_close.pct_change().dropna().loc['2024':]  # 申万1级的收益率

    # 计算协方差矩阵
    normal_cov = sw_l1_rtn.cov()  # 正常的样本估计covariance matrix
    denoised_cov = get_denoised_covariance_matrix(sw_l1_rtn, denoise_method='mean')  # 降噪后的covariance matrix
    detoned_cov = get_detoned_covariance_matrix(denoised_covariance_matrix=denoised_cov)  # detoned后的correlation matrix

    # 计算矩阵的条件数(Condition Number)用来衡量矩阵数值稳定性的指标
    normal_eVal, normal_eVec = getPCA(normal_cov)
    print(f"normal sample covaiance matrix condition num: {np.linalg.cond(normal_cov)}")
    denoised_eVal, denoised_eVec = getPCA(denoised_cov)
    print(f"denoised covaiance matrix condition num: {np.linalg.cond(denoised_cov)}")
    detoned_eVal, detoned_eVec = getPCA(detoned_cov)
    detoned_cov_for_opt = detoned_eVec[:, :-1] @ detoned_eVal[:-1, :-1] @ detoned_eVec[:, :-1].T

    # 可视化协方差矩阵
    fig, (ax_1, ax_2, ax_3) = plt.subplots(1, 3, figsize=(15, 6))
    sns.heatmap(normal_cov.to_numpy(), ax=ax_1)
    sns.heatmap(denoised_cov.to_numpy(), ax=ax_2)
    sns.heatmap(detoned_cov.to_numpy(), ax=ax_3)

    ax_1.set_title("Normal Sample Covariance")
    ax_2.set_title("Denoised Covariance")
    ax_3.set_title("Detoned Covariance")
    plt.show()

####################################################################
# 2. Trend Analysis
####################################################################

def tValLinR(price:pd.Series):
    ''' 
    计算线性趋势beta的t-value:  x_(t+l) = alpha + beta * l + epsilon
    return: beta 的 t-value
    '''
    x = np.ones((price.shape[0],2)) # close行 x 2列的x
    x[:,1] = np.arange(price.shape[0]) # 第二列为时间趋势 l = 0,1,2,3.....price.shape[0]
    ols_result = sm.OLS(price,x).fit() #  拟合回归
    t_value = ols_result.tvalues.iloc[1] # t-value
    return t_value

def getTrendLabel(price_series:pd.Series,l_span:list,disable_tqdm = False):
    ''' 
    根据线性趋势的t值符号标记每个时间点的趋势,测试不同的l_span, 记录t-value绝对值最大的beta的sign作为trend label
    :params price_series: 价格series
    :params l_span: 需要测试的向前看的时间步长list
    :return: pd.DataFrame
    '''
    trend_result_df = pd.DataFrame(index = price_series.index, columns = ['trend_end_date','t_value','label','l','fully_tested'])
    # 遍历每一个date
    for date in tqdm(price_series.index,disable = disable_tqdm):
        date_iloc = price_series.index.get_loc(date) # 日期在index中的位置索引
        trend_result_df_slice = pd.DataFrame(index = l_span, columns = ['trend_end_date','t_value','label'])
        # 如果date + 最大的l 超过了价格序列的长度，那就是没有 fully_tested，需要记录
        if date_iloc + max(l_span) > price_series.shape[0]: 
            fully_tested_flag = 0 
        else:
            fully_tested_flag = 1

        # 测试每一个l
        for l in trend_result_df_slice.index:
            if date_iloc + l > price_series.shape[0]:
                continue
            else:
                trend_end_date = price_series.index[date_iloc + l - 1]
                trend_data_to_test = price_series.iloc[date_iloc:date_iloc + l] # 趋势序列切片
                trend_tvalue = tValLinR(trend_data_to_test) # 计算趋势 t-value
                trend_result_df_slice.loc[l,['trend_end_date','t_value','label']] = trend_end_date, trend_tvalue, np.sign(trend_tvalue) # 记录
        
        trend_result_df_slice = trend_result_df_slice.dropna(axis = 0)
        # 记录这个date的趋势label
        if trend_result_df_slice.empty:
            trend_result_df.loc[date,['trend_end_date','t_value','label','fully_tested']] = np.nan,np.nan,np.nan,fully_tested_flag
        else:
            abs_trend_value = np.abs(trend_result_df_slice['t_value']).astype(float)
            most_significant_tvalue_l = abs_trend_value.idxmax()
            trend_result_df.loc[date] = trend_result_df_slice.loc[most_significant_tvalue_l].values.tolist() + [most_significant_tvalue_l,fully_tested_flag]
        
    return trend_result_df

def get_trend_label_for_plotting(price_series,trend_result):
    trend_label = pd.merge(price_series,trend_result,left_index = True,right_index = True, how = 'left')
    trend_label['normed_t_value'] = trend_label['t_value'] / (2 * trend_label['t_value'].abs().max()) + 0.5
    trend_label = trend_label.dropna()
    return trend_label

####################################################################
# 3. Clustering
####################################################################

def clusterKMeansBase(corr:pd.DataFrame,maxNumClusters = None, n_trials = 10,random_state = 888):
    '''
    基于K-Means聚类 (Base)
    :params corr: 相关性矩阵
    :params maxNumCluster: 最大cluster数量
    :params n_trials 重复次数
    :return corr_reindexed: 根据cluster重新排列后的correlation_matrix
    :return clusters: 聚类信息
    :return shilhouette_series: 轮廓值序列
    '''
    corr_copy = pd.DataFrame(data = np.nan_to_num(corr,nan = 0,copy = True),index = corr.index, columns = corr.columns)
    x = np.sqrt(0.5 * (1 - corr_copy)) # 用来做K-Means的Observation Matrix
    silhouette_series = pd.Series(dtype = float) # 用来储存轮廓值
    kmeans_result = np.nan # KMeans结果

    if not maxNumClusters:
        maxNumClusters = corr.shape[0] - 1

    for trial in range(n_trials):
        for cluster_num in range(2,maxNumClusters + 1):
            kmeans_ = KMeans(n_clusters = cluster_num, n_init = 10,random_state = random_state + trial) # 初始化k-means
            kmeans_result_tmp = kmeans_.fit(x) # 拟合k-means
            silhouette_series_tmp = silhouette_samples(x,kmeans_result_tmp.labels_) # 计算每个样本的轮廓值
            stats = [silhouette_series_tmp.mean()/silhouette_series_tmp.std(), silhouette_series.mean()/silhouette_series.std()] # 评价聚类效果的指标 mean(silhouette) / std(silhouette)
            # stats = [silhouette_series_tmp.mean(), silhouette_series.mean()] # 评价聚类效果的指标 mean(silhouette)
            # print(f"{cluster_num}: {silhouette_series_tmp.mean()/silhouette_series_tmp.std()}")
            if pd.isna(stats[1]) or stats[0] > stats[1]: # 更新评价指标
                silhouette_series,kmeans_result = silhouette_series_tmp,kmeans_result_tmp
    new_idx = np.argsort(kmeans_result.labels_) # 新的排序结果
    corr_reindexed = corr_copy.iloc[new_idx,:] # 重排行
    corr_reindexed = corr_reindexed.iloc[:,new_idx] # 重排列
    # 记录cluster结果
    clusters = {
        cluster_idx:corr_copy.columns[np.where(kmeans_result.labels_ == cluster_idx)[0]].tolist() # np.where返回是一个tuple，所以需要[0]
        for cluster_idx in np.unique(kmeans_result.labels_) 
        }
    # 记录轮廓值
    silhouette_df = pd.DataFrame(index = x.index)
    silhouette_df['Cluster'] = kmeans_result.labels_
    silhouette_df['silhouette_samplevalue'] = silhouette_series
    print(f"Cluster Num: {len(clusters)}")
    return corr_reindexed,clusters,silhouette_df


def makeNewOutputs(corr:pd.DataFrame,clusters_exredo:dict,clusters_redo:dict):
    '''
    合并clusters
    '''
    clustersNew = {} # 新的clusters dict

    for cluster_idx in clusters_exredo.keys(): # 记录原始的
        clustersNew[len(clustersNew.keys())] = list(clusters_exredo[cluster_idx])
    for cluster_idx_redo in clusters_redo.keys(): # 记录redo的
        clustersNew[len(clustersNew.keys())] = list(clusters_redo[cluster_idx_redo])

    # 根据新的cluster排序correlation matrix
    newIdx = []
    for asset_idxlist in clustersNew.values():
        newIdx += asset_idxlist
    corrNew = corr.loc[newIdx,newIdx] # 排序
    corrNew_copy = pd.DataFrame(data = np.nan_to_num(corrNew,nan = 0,copy = True),index = corrNew.index, columns = corrNew.columns)
    x = np.sqrt(0.5 * (1 - corrNew_copy)) # observation matrix 用来计算 silhouette_samples

    kmeans_labels = np.zeros(len(x.columns)) # 新的kmeans cluster 编号
    for cluster_new_idx in clustersNew.keys(): # 记录新编号
        idxs = [x.index.get_loc(k) for k in clustersNew[cluster_new_idx]]
        kmeans_labels[idxs] = cluster_new_idx
    silhouette_dfNew = pd.DataFrame(index = x.index) # 新的轮廓值DataFrame
    silhouette_dfNew['Cluster'] = kmeans_labels
    silhouette_dfNew['silhouette_samplevalue'] = silhouette_samples(x,kmeans_labels)

    return corrNew,clustersNew,silhouette_dfNew

def clusterKMeansTop(corr:pd.DataFrame,maxNumClusters = None, n_trials = 10,random_state = 888):
    '''
    基于K-Means聚类 (Top)
    :params corr: 相关性矩阵
    :params maxNumCluster: 最大cluster数量
    :params n_trials 重复次数
    :return corr_reindexed: 根据cluster重新排列后的correlation_matrix
    :return clusters: 聚类信息
    :return shilhouette_series: 轮廓值序列
    '''
    if not maxNumClusters: 
        maxNumClusters = corr.shape[0] // 2
    corr_reindexed,clusters,silhouette_df = clusterKMeansBase(corr,maxNumClusters,n_trials,random_state) # 全局做一次

    # 计算每个Cluster的t statistic
    cluster_t_stats = {
        cluster_idx:silhouette_df[silhouette_df['Cluster'] == cluster_idx]['silhouette_samplevalue'].mean()
        for cluster_idx in clusters.keys()
        }
    # 计算cluster的t-statistic的均值,少于均值的继续再重新cluster
    tStatsMean = np.sum([cluster_t_stats[cluster_idx] for cluster_idx in cluster_t_stats]) / len(cluster_t_stats) # 计算总体的t均值
    redoClusters = [cluster_idx for cluster_idx in cluster_t_stats.keys() if cluster_t_stats[cluster_idx] < tStatsMean] # cluster的t均值低于`总体cluster的t均值`的cluster需要重新cluster

    # 判断是否有需要重新K-Means聚类的cluster
    if len(redoClusters) < 1:
        print('不存在需要redo的Cluster')
        return corr_reindexed,clusters,silhouette_df # 没有直接返回
    else:
        print('存在需要redo的Clusters')
        clusters_to_redo = {redoCluster_idx:clusters[redoCluster_idx] for redoCluster_idx in redoClusters}
        print(clusters_to_redo)
        keysRedo = [] # 需要redo的asset 
        redotStatsMean = np.mean([cluster_t_stats[redocluster_idx] for redocluster_idx in redoClusters]) # redo的t均值
        for redocluster_idx in redoClusters:
            keysRedo += clusters[redocluster_idx] # 将需要redo的Cluster的asset加入keysRedo
        corr_redo = corr.loc[keysRedo,keysRedo] # 生成Tmp correlation matrix
        corr_reindex_redo,clusters_redo,silhouette_df_redo = clusterKMeansBase(corr_redo,
                                                                                maxNumClusters= min(maxNumClusters,corr_redo.shape[0] // 2),
                                                                                n_trials = n_trials, 
                                                                                random_state = random_state
                                                                                )
        clusters_exredo = {cluster_idx:clusters[cluster_idx] for cluster_idx in clusters.keys() if cluster_idx not in redoClusters} # 不需要redo的cluster
        corrNew,clustersNew,silhouette_dfNew = makeNewOutputs(corr,clusters_exredo,clusters_redo) # 将redo后的和不需要redo的合并
        # 新的cluster_t_stats
        cluster_t_stats_new = {
            cluster_idx:silhouette_dfNew[silhouette_dfNew['Cluster'] == cluster_idx]['silhouette_samplevalue'].mean()
            for cluster_idx in clustersNew.keys()
            }
        
        '''Book Version'''
        newtStatsMean = np.sum([cluster_t_stats_new[cluster_idx_new] for cluster_idx_new in cluster_t_stats_new]) / len(cluster_t_stats_new)
        # print(redotStatsMean,newtStatsMean)
        if newtStatsMean <= redotStatsMean:
            print('redo并没有提升cluster表现')
            return corr_reindexed,clusters,silhouette_df
        else:
            print('redo存在表现提升')
            print(f'New cluster num: {len(clustersNew)}')
            return corrNew,clustersNew,silhouette_dfNew
            
def OptPortWeight(cov,mu = None):
    '''
    计算最优投资组合权重
    :params cov是协方差矩阵
    :params mu是预期收益率向量
    '''
    cov_inv = np.linalg.inv(cov)
    ones = np.ones(shape = (cov_inv.shape[0],1))
    if not mu:
        mu = ones
    w_raw = cov_inv @ mu
    w_sum = ones.T @ w_raw
    w = w_raw / w_sum
    return w

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

def get_order(linkage_matrix, n_items):
    """
    获取层次聚类的排序
    
    参数:
        linkage_matrix: 层次聚类连接矩阵
        n_items: 项目数量
        
    返回:
        聚类排序
    """
    # 创建编号
    order = []
    for i in range(n_items):
        order.append([i])
    for i in range(linkage_matrix.shape[0]): # 遍历每一次的合并信息
        cluster1 = int(linkage_matrix[i, 0]) # 第一个簇
        cluster2 = int(linkage_matrix[i, 1]) # 第二个簇
        order.append(order[cluster1] + order[cluster2]) # 添加新的簇
    return order[-1] # 返回最后合并的结果

def clusterSpectralClusteringBase(corr:pd.DataFrame,maxNumClusters = None, n_trials = 10,random_state = 888):
    '''
    基于Spectral Clustering聚类 (Base) Scikit-Learn 实现的是NCut
    
    参数:
        corr: 相关性矩阵
        maxNumCluster: 最大cluster数量
        n_trials: 重复次数
        random_state: 随机种子
        
    返回:
        corr_reindexed: 根据cluster重新排列后的correlation_matrix
        clusters: 聚类信息
        shilhouette_series: 轮廓值序列
    '''
    from sklearn.cluster import SpectralClustering
    
    corr_copy = pd.DataFrame(data = np.nan_to_num(corr,nan = 0,copy = True),index = corr.index, columns = corr.columns)
    x = np.sqrt(0.5 * (1 - corr_copy)) # 用来做K-Means的Observation Matrix
    silhouette_series = pd.Series(dtype = float) # 用来储存轮廓值
    spectralcluster_result = np.nan    
    
    if not maxNumClusters:
        maxNumClusters = corr.shape[0] - 1
    for trial in range(n_trials):
        for cluster_num in range(2,maxNumClusters + 1):
            spectralcluster_ = SpectralClustering(n_clusters = cluster_num, n_init = 10,affinity = 'precomputed',random_state = random_state + trial)
            spectralcluster_result_tmp = spectralcluster_.fit(x) # 拟合k-means
            silhouette_series_tmp = silhouette_samples(x,spectralcluster_result_tmp.labels_) # 计算每个样本的轮廓值
            stats = [silhouette_series_tmp.mean()/silhouette_series_tmp.std(), silhouette_series.mean()/silhouette_series.std()] # 评价聚类效果的指标 mean(silhouette) / std(silhouette)
            # stats = [silhouette_series_tmp.mean(), silhouette_series.mean()] # 评价聚类效果的指标 mean(silhouette)
            # print(f"{cluster_num}: {silhouette_series_tmp.mean()/silhouette_series_tmp.std()}")
            if pd.isna(stats[1]) or stats[0] > stats[1]: # 更新评价指标
                silhouette_series,spectralcluster_result = silhouette_series_tmp,spectralcluster_result_tmp
    new_idx = np.argsort(spectralcluster_result.labels_) # 新的排序结果
    corr_reindexed = corr_copy.iloc[new_idx,:] # 重排行
    corr_reindexed = corr_reindexed.iloc[:,new_idx] # 重排列
    # 记录cluster结果
    clusters = {
        cluster_idx:corr_copy.columns[np.where(spectralcluster_result.labels_ == cluster_idx)[0]].tolist() # np.where返回是一个tuple，所以需要[0]
        for cluster_idx in np.unique(spectralcluster_result.labels_) 
        }
    # 记录轮廓值
    silhouette_df = pd.DataFrame(index = x.index)
    silhouette_df['Cluster'] = spectralcluster_result.labels_
    silhouette_df['silhouette_samplevalue'] = silhouette_series
    print(f"Cluster Num: {len(clusters)}")
    return corr_reindexed,clusters,silhouette_df

def Signed_Laplacian_Clustering(affinity_matrix, n_clusters, method='symmetric_normalized', random_state=100):
    '''
    Signed Laplacian Clustering
    
    参数:
        affinity_matrix: A = affinity_matrix. D_ii = Σ |A_ij|.
        n_clusters: 聚类数
        method: 
            "symmetric_normalized": L_sym = I - D^(-1/2) A D^(-1/2)
            "random_walk": L_rw = I - D^(-1) A
        random_state: 随机种子
            
    返回:
        labels: 聚类结果
    '''
    
    A = affinity_matrix.copy() # Affinity Matrix
    D = np.diag(np.sum(np.abs(A), axis=1))  # Degree Matrix
    I = np.eye(D.shape[0])  # Unit Matrix
    
    if method == 'symmetric_normalized':
        # symmetric normalize laplacian matrix
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
        L = I - D_inv_sqrt @ A @ D_inv_sqrt
    elif method == 'random_walk':
        # random walk normalize laplacian matrix
        D_inv = np.diag(1.0 / np.diag(D))
        L = I - D_inv @ A
    else:
        raise ValueError("method must be 'symmetric_normalized' or 'random_walk'")
    
    # compute eigendecomposition of L
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    # sort eigenvalues and eigenvectors
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # select first k eigenvectors
    selected_eigenvectors = eigenvectors[:, :n_clusters]
    
    # cluster with k-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(selected_eigenvectors)
    
    return labels

from scipy.sparse.linalg import lobpcg

def SPONGE(affinity_matrix, n_clusters, tau_plus, tau_minus, symmetric_normalize=False, random_state=100):
    '''
    Signed Positive Over Negative Generalized Eigenproblem (SPONGE)
    
    参数:
        affinity_matrix: 带符号的相似度矩阵 (A = A_plus - A_minus)
        n_clusters: 聚类数量
        tau_plus: 正向互动的正则化参数 (必须 > 0)
        tau_minus: 负向互动的正则化参数 (必须 > 0)
        symmetric_normalize: 是否应用对称规范化 (默认: False)
        random_state: K-Means的随机种子 (默认: 100)
        
    返回:
        labels: 聚类分配结果
    '''
    if tau_plus <= 0 or tau_minus <= 0:
        raise ValueError(f"tau_plus = {tau_plus}, tau_minus = {tau_minus}. Both must be positive.")
    
    A = affinity_matrix.copy()
    
    # Split into positive and negative parts
    A_plus = np.where(A > 0, A, 0)
    A_minus = np.where(A < 0, -A, 0)  # Use -A to ensure A_minus is non-negative
    
    # Compute degree matrices
    D_plus = np.diag(np.sum(A_plus, axis=1))
    D_minus = np.diag(np.sum(A_minus, axis=1))
    
    # Compute Laplacians
    L_plus = D_plus - A_plus
    L_minus = D_minus - A_minus
    
    if symmetric_normalize:
        # Symmetric normalization
        D_plus_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D_plus) + 1e-10))
        D_minus_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D_minus) + 1e-10))
        
        L_plus_sym = D_plus_inv_sqrt @ L_plus @ D_plus_inv_sqrt
        L_minus_sym = D_minus_inv_sqrt @ L_minus @ D_minus_inv_sqrt
        
        # Regularize
        I = np.eye(A.shape[0])
        L_plus_normed = L_plus_sym + tau_minus * I
        L_minus_normed = L_minus_sym + tau_plus * I
    else:
        # Unnormalized version
        L_plus_normed = L_plus + tau_minus * D_minus
        L_minus_normed = L_minus + tau_plus * D_plus
    
    # Initial guess for eigenvectors (random initialization)
    X = np.random.rand(A.shape[0], n_clusters)
    
    # Solve generalized eigenvalue problem: L_plus_normed v = λ L_minus_normed v
    eigenvalues, eigenvectors = lobpcg(
        A=L_plus_normed,
        X=X,
        B=L_minus_normed,
        largest=False  # Find smallest eigenvalues
    )
    
    # Cluster the eigenvectors using K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(eigenvectors)
    
    return labels 

####################################################################
# 4. Feature Importance
####################################################################

def getTestData(n_features=100, n_informative=25, n_redundant=25, n_samples=10000, random_state=0, sigmaStd=0.0):
    """
    生成用于分类问题的随机数据集
    
    参数:
        n_features: 特征数量
        n_informative: 信息特征数量
        n_redundant: 冗余特征数量
        n_samples: 样本数量
        random_state: 随机种子
        sigmaStd: 噪声标准差
        
    返回:
        X, y: 特征矩阵和标签
    """
    # generate a random dataset for a classification problem
    np.random.seed(random_state)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features-n_redundant,
        n_informative=n_informative,
        n_redundant=0,
        shuffle=False,
        random_state=random_state
    )
    cols = ['I_'+str(i) for i in range(n_informative)]
    cols += ['N_'+str(i) for i in range(n_features-n_informative-n_redundant)]
    X, y = pd.DataFrame(X, columns=cols), pd.Series(y)
    i = np.random.choice(range(n_informative), size=n_redundant)
    for k, j in enumerate(i):
        X['R_'+str(k)] = X['I_'+str(j)] + np.random.normal(size=X.shape[0]) * sigmaStd
    return X, y

def feature_impurity_MDI(fit_result, feature_names):
    """
    计算基于每个特征的 Mean Decrease Impurity
    
    优点:
        (1). 没有强分布假设
        (2). 方差减少(合并更多估计器)
        (3). 重要性基于概率而非代数方程估计
    缺点:
        (1). 样本内显著的变量可能与样本外无关
        
    参数:
        fit_result: bagging model 的拟合结果
        feature_names: 特征的名称
        
    返回:
        feature_MDI_result_df: 特征的Mean Decrease Impurity DataFrame
    """
    tree_feature_importance_df = pd.DataFrame(columns=feature_names) # 用来储存每个学习器的feature importance
    for tree_idx, tree in enumerate(fit_result.estimators_): # 遍历每一个estimator, 记录feature importance
        tree_feature_importance_df.loc[tree_idx] = tree.feature_importances_ # mean descrease impurity
    tree_feature_importance_df = tree_feature_importance_df.replace(0, np.nan) 
    tree_feature_importance_mean = tree_feature_importance_df.mean(axis=0).to_frame(name='mean') # 均值
    tree_feature_importance_mean /= tree_feature_importance_mean.sum() # 归一化
    tree_feature_importance_se = tree_feature_importance_df.std(axis=0).to_frame(name='standard error') / np.sqrt(tree_feature_importance_df.shape[0]) # 标准误
    feature_MDI_result_df = pd.concat([tree_feature_importance_mean, tree_feature_importance_se], axis=1).sort_values(by='mean', ascending=False) # 合并
    return feature_MDI_result_df

def plot_feature_importance(feature_importance_df, figsize=(6, 4), title='Feature Importance', ylabel=None, 
                           title_fontsize=12, xlabel_fontszie=12, xtick_rotations=45, ylabel_fontsize=12):
    """
    绘制特征重要性图表
    
    参数:
        feature_importance_df: 特征重要性DataFrame
        figsize: 图表大小
        title: 图表标题
        ylabel: Y轴标签
        title_fontsize: 标题字体大小
        xlabel_fontszie: X轴标签字体大小
        xtick_rotations: X轴标签旋转角度
        ylabel_fontsize: Y轴标签字体大小
    """
    plt.figure(figsize=figsize)
    bars = plt.bar(
        x=feature_importance_df.index,         # X轴标签
        height=feature_importance_df['mean'],  # 柱高
        yerr=feature_importance_df['standard error'],     # 误差线
        capsize=5,          # 误差线顶部横杠长度
        color='steelblue',  # 柱颜色
        edgecolor='black'   # 边框颜色
    )
    # 添加注释
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')

    plt.title(title, fontsize=title_fontsize)
    plt.xlabel('Features', fontsize=xlabel_fontszie)
    plt.ylabel(ylabel, fontsize=ylabel_fontsize)
    plt.xticks(rotation=xtick_rotations)  # 旋转X轴标签
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # 添加横向网格线
    plt.tight_layout()
    plt.show()

def probability_weight_accuracy(y_true, y_pred, labels):
    """
    计算概率加权准确率 (Probability Weighted Accuracy)
    
    PWA比准确率更严厉地惩罚高置信度的错误预测，但没有对数损失那么严厉
    
    参数:
        y_true: 真实标签
        y_pred: 预测概率
        labels: 标签列表
        
    返回:
        PWA值
    """
    y_pred_max = y_pred.max(axis=1).reshape(-1, 1) # 每一个sample给出不同标签的预测概率，取最大值
    y_pred_label = labels[np.argmax(y_pred, axis=1)] # 每一个sample的预测标签
    y_correct_flag = (y_true == y_pred_label).astype(int).to_numpy().reshape(-1, 1) # 是否预测正确
    K = labels.shape[0] # 标签数量
    epsilon = 1e-10  # 避免除零
    PWA = (y_correct_flag * (y_pred_max - 1/K)).sum() / ((y_pred_max - 1/K).sum() + epsilon)
    return PWA

def feature_impurity_MDA(bagging_estimator, X, y, n_splits=5, eval_metric=log_loss, eval_metric_higher_better=False, disable_tqdm=False):
    """
    计算基于每个特征的 Mean Decrease Accuracy
    
    如果一个特征很重要，那么打乱这个特征应该会导致性能显著下降
    log-loss = - (y * log(p) + (1-y)log(1-p))。对数损失的值越小，表示模型的预测越准确。
    这样同时考虑了 'proportion' 和 'probability'
    
    优点:
        (1). 缓解了MDI的"样本外有效性"问题，只要特征是独立的
    缺点:
        (1). 如果有多个相互关联的特征，那么打乱一个特征的数据可能会被不打乱其他特征所补偿
        
    参数:
        bagging_estimator: bagging模型估计器
        X: 特征矩阵
        y: 因变量序列
        n_splits: KFold测试的次数
        eval_metric: 评价标准，默认是log_loss
        eval_metric_higher_better: 评价指标是否越大越好
        disable_tqdm: 是否禁用进度条
        
    返回:
        feature_MDA_result_df: 特征的Mean Decrease Accuracy DataFrame
    """
    if eval_metric_higher_better: # 是否越大越好,如果不是的话 * -1 变成越大越好
        eval_metric_multiplier = 1
    else:
        eval_metric_multiplier = -1
    cross_validation = KFold(n_splits=n_splits, shuffle=False) # 交叉验证器
    score_original = pd.Series(dtype=float) # 记录原始的log-loss
    score_shuffle = pd.DataFrame(columns=X.columns) # 储存打乱后的pred prob log-loss
    
    for validate_idx, (train_indices, test_indices) in tqdm(enumerate(cross_validation.split(X=X)), total=n_splits, desc='Calculating MDA...', disable=disable_tqdm):
        X_train, y_train = X.iloc[train_indices, :], y.iloc[train_indices] # 训练集
        X_test, y_test = X.iloc[test_indices, :], y.iloc[test_indices] # 测试集
        fit_result = bagging_estimator.fit(X=X_train, y=y_train) # 拟合
        pred_prob = fit_result.predict_proba(X_test) # 预测test的prob before shuffling
        
        # 计算原始分数
        eval_metric_dict = {'y_true': y_test, 'y_pred': pred_prob, 'labels': bagging_estimator.classes_}
        score_original.loc[validate_idx] = eval_metric_multiplier * eval_metric(**eval_metric_dict)
        
        # 遍历所有特征,打乱特征的value
        for feature_n in score_shuffle.columns:
            X_test_shuffle = X_test.copy(deep=True) # 复制一份测试集
            feature_values = X_test_shuffle[feature_n].to_numpy()
            np.random.shuffle(feature_values)
            X_test_shuffle.loc[:, feature_n] = feature_values
            pred_prob_shuffled = fit_result.predict_proba(X_test_shuffle) # 预测
            eval_metric_dict_shuffled = {'y_true': y_test, 'y_pred': pred_prob_shuffled, 'labels': bagging_estimator.classes_}
            score_shuffle.loc[validate_idx, feature_n] = eval_metric_multiplier * eval_metric(**eval_metric_dict_shuffled)
            
    epsilon = 1e-12
    score_change_pct = (score_shuffle - score_original.to_numpy().reshape(-1, 1)) / (np.abs(score_original.to_numpy().reshape(-1, 1)) + epsilon) # 变化的百分比
    feature_decrease_accuracy_mean = score_change_pct.mean(axis=0).to_frame(name='mean') * (-1) # * (-1)是为了调整，这样结果就是mean越大，重要性越高
    feature_descreae_accuarcy_se = score_change_pct.std(axis=0).to_frame(name='standard error') / np.sqrt(score_change_pct.shape[0])
    feature_MDA_result_df = pd.concat([feature_decrease_accuracy_mean, feature_descreae_accuarcy_se], axis=1).sort_values(by='mean', ascending=False) # 合并
    return feature_MDA_result_df

def groupMeanStd(feature_importance_df, clstrs):
    """
    将聚类中一个类的特征重要性合并
    
    参数:
        feature_importance_df: 特征重要性DataFrame
        clstrs: 聚类结果字典
        
    返回:
        聚类特征重要性DataFrame
    """
    out = pd.DataFrame(columns=['mean', 'std'])
    for cluster_idx, cluster_components in clstrs.items():
        group_sum = feature_importance_df[cluster_components].sum(axis=1)
        out.loc['C_'+str(cluster_idx), 'mean'] = group_sum.mean()  # 以C开头代表cluster
        out.loc['C_'+str(cluster_idx), 'standard error'] = group_sum.std() / np.sqrt(group_sum.shape[0])
    return out

def feature_impurity_MDI_Clustered(fit_result, feature_names, clstrs):
    """
    计算基于Cluster的Cluster Feature MDI
    
    参数:
        fit_result: bagging model的拟合结果
        feature_names: 特征的名称
        clstrs: 特征聚类结果字典，由clusterKMeansBase或clusterKMeansTop产出的cluster结果
        
    返回:
        feature_MDI_cluster_result_df: 聚类特征的Mean Decrease Impurity DataFrame
    """
    tree_feature_importance_df = pd.DataFrame(columns=feature_names) # 用来储存每个学习器的feature importance
    for tree_idx, tree in enumerate(fit_result.estimators_): # 遍历每一个estimator, 记录feature importance
        tree_feature_importance_df.loc[tree_idx] = tree.feature_importances_ # mean descrease impurity
    tree_feature_importance_df = tree_feature_importance_df.replace(0, np.nan) 
    feature_MDI_cluster_result_df = groupMeanStd(tree_feature_importance_df, clstrs)
    feature_MDI_cluster_result_df['mean'] = feature_MDI_cluster_result_df['mean'] / feature_MDI_cluster_result_df['mean'].sum() # 均值标准化
    return feature_MDI_cluster_result_df.sort_values(by='mean', ascending=False)

def feature_impurity_MDA_Clustered(bagging_estimator, X, y, clsters, n_splits=5, eval_metric=log_loss, eval_metric_higher_better=False, disable_tqdm=False):
    """
    计算基于每个Cluster的特征 Mean Decrease Accuracy
    
    参数:
        bagging_estimator: bagging模型估计器
        X: 特征矩阵
        y: 因变量序列
        clsters: 特征聚类结果字典，由clusterKMeansBase或clusterKMeansTop产出的cluster结果
        n_splits: KFold测试的次数
        eval_metric: 评价标准，默认是log_loss
        eval_metric_higher_better: 评价指标是否越大越好
        disable_tqdm: 是否禁用进度条
        
    返回:
        feature_MDA_result_df: 聚类特征的Mean Decrease Accuracy DataFrame
    """
    if eval_metric_higher_better: # 是否越大越好,如果不是的话 * -1 变成越大越好
        eval_metric_multiplier = 1
    else:
        eval_metric_multiplier = -1
    cross_validation = KFold(n_splits=n_splits, shuffle=False) # 交叉验证器
    score_original = pd.Series(dtype=float) # 记录原始的log-loss
    score_shuffle = pd.DataFrame(columns=list(clsters.keys())) # 储存打乱后的pred prob log-loss
    
    for validate_idx, (train_indices, test_indices) in tqdm(enumerate(cross_validation.split(X=X)), total=n_splits, desc='Calculating Cluster MDA...', disable=disable_tqdm):
        X_train, y_train = X.iloc[train_indices, :], y.iloc[train_indices] # 训练集
        X_test, y_test = X.iloc[test_indices, :], y.iloc[test_indices] # 测试集
        fit_result = bagging_estimator.fit(X=X_train, y=y_train) # 拟合
        pred_prob = fit_result.predict_proba(X_test) # 预测test的prob before shuffling
        eval_metric_dict = {'y_true': y_test, 'y_pred': pred_prob, 'labels': bagging_estimator.classes_}
        score_original.loc[validate_idx] = eval_metric_multiplier * eval_metric(**eval_metric_dict)
        
        # 遍历所有特征,打乱特征的value
        for cluster_idx in score_shuffle.columns:
            X_test_shuffle = X_test.copy(deep=True) # 复制一份测试集
            for cluster_component in clsters[cluster_idx]: # 对于cluster中的每个资产，打乱
                feature_values = X_test_shuffle[cluster_component].to_numpy()
                np.random.shuffle(feature_values)
                X_test_shuffle.loc[:, cluster_component] = feature_values
            pred_prob_shuffled = fit_result.predict_proba(X_test_shuffle) # 预测
            eval_metric_dict_shuffled = {'y_true': y_test, 'y_pred': pred_prob_shuffled, 'labels': bagging_estimator.classes_}
            score_shuffle.loc[validate_idx, cluster_idx] = eval_metric_multiplier * eval_metric(**eval_metric_dict_shuffled)
            
    epsilon = 1e-12
    score_change_pct = (score_shuffle - score_original.to_numpy().reshape(-1, 1)) / (np.abs(score_original.to_numpy().reshape(-1, 1)) + epsilon) # 变化的百分比
    feature_decrease_accuracy_mean = score_change_pct.mean(axis=0).to_frame(name='mean') * (-1) # * (-1)是为了调整，这样结果就是mean越大，重要性越高
    feature_descreae_accuarcy_se = score_change_pct.std(axis=0).to_frame(name='standard error') / np.sqrt(score_change_pct.shape[0])
    feature_MDA_result_df = pd.concat([feature_decrease_accuracy_mean, feature_descreae_accuarcy_se], axis=1).sort_values(by='mean', ascending=False) # 合并
    feature_MDA_result_df.index = [f"C_{cluster_idx}" for cluster_idx in feature_MDA_result_df.index]
    return feature_MDA_result_df 

####################################################################
# 5. Information Theory
####################################################################

def OptNumBins(n_obs, corr_coef=False):
    """
    决定离散化的最优bin数量
    
    参数:
        n_obs: 观测值数量
        corr_coef: 是否为联合熵情况（默认为False）
        
    返回:
        最优bin数量
    """
    if corr_coef: # joint entropy case <= Hacine-Gharbi and Ravier (2018)
        opt_bins = round(1 / np.sqrt(2) * np.sqrt(1 + np.sqrt(1 + (24 * n_obs) / (1 - corr_coef ** 2))))
    else: # marginal entropy case <= Hacine-Gharbi et al. (2012)
        z = (8 + 324 * n_obs + 12 * np.sqrt(36 * n_obs + 729 * n_obs**2))**(1/3)
        opt_bins = round(z / 6 + 2 / (3 * z) + 1 / 3)
    return int(opt_bins)

def joint_entropy(x, y):
    """
    计算x,y的联合熵
    
    参数:
        x, y: 输入变量
        
    返回:
        联合熵H(x,y)
    """
    bins = OptNumBins(x.shape[0], corr_coef=np.corrcoef(x, y)[0, 1]) # 选择最优bins数量
    contingency_xy = np.histogram2d(x=x, y=y, bins=bins)[0] # xy的相依关系
    I_xy = mutual_info_score(labels_true=None, labels_pred=None, contingency=contingency_xy) # xy的mutual information
    H_x = ss.entropy(np.histogram(x, bins)[0]) # x的entropy
    H_y = ss.entropy(np.histogram(y, bins)[0]) # y的entropy
    H_xy = H_x + H_y - I_xy 
    return H_xy

def conditional_entropy(x, y):
    """
    计算条件熵 H(x|y)
    
    参数:
        x, y: 输入变量
        
    返回:
        条件熵H(x|y)
    """
    bins = OptNumBins(x.shape[0], corr_coef=np.corrcoef(x, y)[0, 1]) # 选择最优bins数量
    contingency_xy = np.histogram2d(x=x, y=y, bins=bins)[0] # xy的相依关系
    I_xy = mutual_info_score(labels_true=None, labels_pred=None, contingency=contingency_xy) # xy的mutual information
    H_x = ss.entropy(np.histogram(x, bins)[0]) # x的entropy
    H_y = ss.entropy(np.histogram(y, bins)[0]) # y的entropy
    Hx_y = H_x - I_xy
    return Hx_y

def mutual_information(x, y, norm=False):
    """
    计算x,y的互信息
    
    参数:
        x, y: 输入变量
        norm: 是否归一化（默认为False）
        
    返回:
        互信息I(x;y)
    """
    bins = OptNumBins(x.shape[0], corr_coef=np.corrcoef(x, y)[0, 1]) # 选择最优bins数量
    contingency_xy = np.histogram2d(x=x, y=y, bins=bins)[0] # xy的相依关系
    I_xy = mutual_info_score(labels_true=None, labels_pred=None, contingency=contingency_xy) # xy的mutual information
    if norm: # 由于 0 ≤ I_xy ≤ min(H_x, H_y), 所以可以norm到0-1之间
        H_x = ss.entropy(np.histogram(x, bins)[0]) # x的entropy
        H_y = ss.entropy(np.histogram(y, bins)[0]) # y的entropy
        I_xy /= min(H_x, H_y)
    return I_xy

def variation_information(x, y, norm=False, sharper_norm=False):
    """
    计算x,y的变异信息
    
    变异信息衡量了如果我们知道一个变量的值，我们对另一个变量的期望不确定性
    
    参数:
        x, y: 输入变量
        norm: 是否归一化（默认为False）
        sharper_norm: 是否使用更锐利的归一化（默认为False）
        
    返回:
        变异信息VI(x,y)
    """
    bins = OptNumBins(x.shape[0], corr_coef=np.corrcoef(x, y)[0, 1]) # 选择最优bins数量
    contingency_xy = np.histogram2d(x=x, y=y, bins=bins)[0] # xy的相依关系
    I_xy = mutual_info_score(labels_true=None, labels_pred=None, contingency=contingency_xy) # xy的mutual information
    H_x = ss.entropy(np.histogram(x, bins)[0]) # x的entropy
    H_y = ss.entropy(np.histogram(y, bins)[0]) # y的entropy
    variation_information_xy = H_x + H_y - 2 * I_xy # xy的variation information 

    if norm: # norm之后将variation information bounded between 0 - 1
        if not sharper_norm:
            H_xy = H_x + H_y - I_xy
            variation_information_xy /= H_xy
        else: # Kraskov et al. (2008)
            variation_information_xy = 1 - I_xy / max(H_x, H_y)
    
    return variation_information_xy 