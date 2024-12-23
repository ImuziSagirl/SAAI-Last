import torch
import torch.nn.functional as F
import numpy as np


# def getNewFeature(x, y, k1, k2, mean: bool = False):
#     dismat = x @ y.T
#     val, rank = dismat.topk(k1)
#     dismat[dismat < val[:, -1].unsqueeze(1)] = 0
#     if mean:
#         dismat = dismat[rank[:, :k2]].mean(dim=1)
#     return dismat


# def AIM(qf: torch.tensor, gf: torch.tensor, k1, k2):
#     qf = qf.to('cuda')
#     gf = gf.to('cuda')

#     qf = torch.nn.functional.normalize(qf)
#     gf = torch.nn.functional.normalize(gf)

#     new_qf = torch.concat([getNewFeature(qf, gf, k1, k2)], dim=1)
#     new_gf = torch.concat([getNewFeature(gf, gf, k1, k2, mean=True)], dim=1)

#     new_qf = torch.nn.functional.normalize(new_qf)
#     new_gf = torch.nn.functional.normalize(new_gf)

#     # additional use of relationships between query sets
#     # new_qf = torch.concat([getNewFeature(qf, qf, k1, k2, mean=True), getNewFeature(qf, gf, k1, k2)], dim=1)
#     # new_gf = torch.concat([getNewFeature(gf, qf, k1, k2), getNewFeature(gf, gf, k1, k2, mean=True)], dim=1)

#     return (-new_qf @ new_gf.T - qf @ gf.T).to('cpu')




def getNewFeature(x, y, k1, k2, mean: bool = False):

    # 计算相似度矩阵
    dismat = x @ y.T

    # 噪声抑制
    val, rank = dismat.topk(k1)
    dismat[dismat < val[:, -1].unsqueeze(1)] = 0

    # 特征扩展
    if mean:
        # 取top k2个值并求均值
        dismat = dismat[rank[:, :k2]].mean(dim=1)

    return dismat


def AIM_RegDB(qf: torch.tensor, gf: torch.tensor, k1, k2):
    # 移动到GPU并归一化
    qf = qf.to('cuda')
    gf = gf.to('cuda')
    qf = F.normalize(qf, p=2, dim=1)
    gf = F.normalize(gf, p=2, dim=1)

    # 多关系特征建模
    new_qf = torch.concat([
        getNewFeature(qf, gf, k1, k2),  # 查询-图库关系            3803 301                     2060 2060
        getNewFeature(gf, gf, k1, k2,mean=True),  # 查询-图库关系  301 301                      2060 2060
    ], dim=1)
    # new_qf = getNewFeature(qf, gf, k1, k2)+getNewFeature(gf, gf, k1, k2, mean=True)
    new_gf = torch.concat([
        getNewFeature(gf, gf, k1, k2, mean=True),  # 图库内部关系  301 301                      2060 2060
        getNewFeature(gf, qf, k1, k2)  # 图库-查询关系             301 3803                     2060 2060
    ], dim=1)
    # new_gf = getNewFeature(gf, gf, k1, k2, mean=True) + getNewFeature(gf, qf, k1, k2)

    # 归一化新特征
    new_qf = F.normalize(new_qf, p=2, dim=1)
    new_gf = F.normalize(new_gf, p=2, dim=1)

    # 计算原始距离和AIM距离
    origin_dist = -qf @ gf.T
    aim_dist = -new_qf @ new_gf.T

    # 自适应权重融合
    alpha = 1  # 可通过交叉验证调整
    final_dist = alpha * aim_dist + (1 - alpha) * origin_dist

    return final_dist.to('cpu')

def AIM_SYSU(qf: torch.tensor, gf: torch.tensor, k1, k2):
    # 移动到GPU并归一化
    qf = qf.to('cuda')
    gf = gf.to('cuda')
    qf = F.normalize(qf, p=2, dim=1)
    gf = F.normalize(gf, p=2, dim=1)

    # 多关系特征建模
    new_qf = torch.concat([
        getNewFeature(qf, gf, k1, k2),  # 查询-图库关系            3803 301                     2060 2060
        #getNewFeature(gf, gf, k1, k2,mean=True),  # 查询-图库关系  301 301                      2060 2060
    ], dim=1)
    # new_qf = getNewFeature(qf, gf, k1, k2)+getNewFeature(gf, gf, k1, k2, mean=True)
    new_gf = torch.concat([
        getNewFeature(gf, gf, k1, k2, mean=True),  # 图库内部关系  301 301                      2060 2060
        #etNewFeature(gf, qf, k1, k2)  # 图库-查询关系             301 3803                     2060 2060
    ], dim=1)
    # new_gf = getNewFeature(gf, gf, k1, k2, mean=True) + getNewFeature(gf, qf, k1, k2)

    # 归一化新特征
    new_qf = F.normalize(new_qf, p=2, dim=1)
    new_gf = F.normalize(new_gf, p=2, dim=1)

    # 计算原始距离和AIM距离
    origin_dist = -qf @ gf.T
    aim_dist = -new_qf @ new_gf.T

    # 自适应权重融合
    alpha = 1  # 可通过交叉验证调整
    final_dist = alpha * aim_dist + (1 - alpha) * origin_dist

    return final_dist.to('cpu')


# import torch
# import torch.nn.functional as F
# from sklearn.decomposition import PCA
# import numpy as np

# def getNewFeature(x, y, k1, k2, mean: bool = False):

#     # 计算相似度矩阵
#     dismat = x @ y.T

#     # 噪声抑制
#     val, rank = dismat.topk(k1)
#     dismat[dismat < val[:, -1].unsqueeze(1)] = 0

#     # 特征扩展
#     if mean:
#         # 取top k2个值并求均值
#         dismat = dismat[rank[:, :k2]].mean(dim=1)

#     return dismat

# def pca_reduction(features, target_samples=301):
#     """
#     仅对SYSU数据集的查询集特征进行PCA降维
    
#     Args:
#     - features: 原始特征
#     - target_samples: 目标样本数，默认为301
    
#     Returns:
#     - 降维后的特征
#     """
#     # 检查是否需要降维（SYSU数据集查询集特征shape为[3803,16384]）
#     if features.shape[0] == 3803:
#         if isinstance(features, torch.Tensor):
#             features_np = features.cpu().numpy()
#         else:
#             features_np = features
        
#         # 初始化PCA
#         pca = PCA(n_components=target_samples)
        
#         # 拟合并转换
#         reduced_features = pca.fit_transform(features_np.T).T  # 注意这里需要转置
        
#         # 转回torch tensor
#         return torch.tensor(reduced_features, dtype=torch.float32)
    
#     # 对于RegDB数据集或SYSU的图库集，直接返回原特征
#     return features

# def AIM(qf: torch.tensor, gf: torch.tensor, k1, k2):
#     # 仅对SYSU数据集的查询集特征降维
#     if qf.shape[0] == 3803:  # SYSU数据集的查询集
#         qf_processed = pca_reduction(qf)
#     else:  # RegDB数据集
#         qf_processed = qf
    
#     # 图库集保持不变
#     gf_processed = gf
    
#     # 移动到GPU并归一化
#     qf_processed = qf_processed.to('cuda')
#     gf_processed = gf_processed.to('cuda')
#     qf_processed = F.normalize(qf_processed, p=2, dim=1)
#     gf_processed = F.normalize(gf_processed, p=2, dim=1)
    
#     # 多关系特征建模
#     new_qf = torch.concat([
#         getNewFeature(qf_processed, gf_processed, k1, k2),  # 查询-图库关系
#         getNewFeature(gf_processed, gf_processed, k1, k2, mean=True),  # 查询-图库关系
#     ], dim=1)
    
#     new_gf = torch.concat([
#         getNewFeature(gf_processed, gf_processed, k1, k2, mean=True),  # 图库内部关系
#         getNewFeature(gf_processed, qf_processed, k1, k2)  # 图库-查询关系
#     ], dim=1)
    
#     # 归一化新特征
#     new_qf = F.normalize(new_qf, p=2, dim=1)
#     new_gf = F.normalize(new_gf, p=2, dim=1)
    
#     # 计算原始距离和AIM距离
#     origin_dist = -qf_processed @ gf_processed.T
#     aim_dist = -new_qf @ new_gf.T
    
#     # 自适应权重融合
#     alpha = 1  # 可通过交叉验证调整
#     final_dist = alpha * aim_dist + (1 - alpha) * origin_dist
    
#     return final_dist.to('cpu')

# 使用示例
# RegDB数据集：
# final_dist = AIM_with_conditional_PCA(query_features, gallery_features, k1=10, k2=5)
# 
# SYSU数据集：
# final_dist = AIM_with_conditional_PCA(query_features, gallery_features, k1=10, k2=5)