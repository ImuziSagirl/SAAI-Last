import torch
import torch.nn.functional as F

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
    print("dismat")
    print(dismat.shape)
    return dismat


def expand_matrix(matrix, target_rows):
    """
    将小矩阵扩展到目标行数
    
    Args:
    - matrix: 原始矩阵，形状为 [n, n]
    - target_rows: 目标行数
    
    Returns:
    - 扩展后的矩阵，形状为 [target_rows, n]
    """
    n = matrix.shape[0]
    expanded_matrix = matrix.unsqueeze(0).expand(target_rows, -1, -1)[:, :, 0]
    return expanded_matrix

def AIM(qf: torch.tensor, gf: torch.tensor, k1, k2):
    # 移动到GPU并归一化
    qf = qf.to('cuda')
    gf = gf.to('cuda')
    qf = F.normalize(qf, p=2, dim=1)
    gf = F.normalize(gf, p=2, dim=1)
    
    # 多关系特征建模
    # 针对不同数据集的特征维度处理
    if qf.shape[0] != gf.shape[0]:  # 针对SYSU数据集
        # 查询-图库关系
        qg_relation = getNewFeature(qf, gf, k1, k2)  # [m, n]
        
        # 图库-图库关系
        gg_relation = getNewFeature(gf, gf, k1, k2, mean=True)  # [n, n]
        
        # 将图库-图库关系扩展到与查询-图库关系相同的行数
        gg_relation_expanded = expand_matrix(gg_relation, qg_relation.shape[0])  # [m, n]
        
        new_qf = torch.concat([
            qg_relation,  # [m, n]
            gg_relation_expanded  # [m, n]
        ], dim=1)  # 结果将是 [m, 2n]
        
        # 图库-查询关系
        gq_relation = getNewFeature(gf, qf, k1, k2)  # [n, m]
        
        # 将图库-图库关系扩展到与图库-查询关系相同的列数
        gg_relation_expanded = expand_matrix(gg_relation, gq_relation.shape[1]).T  # [m, n]
        
        new_gf = torch.concat([
            gg_relation_expanded,  # [m, n]
            gq_relation.T  # [m, n]
        ], dim=1)  # 结果将是 [m, 2n]
    else:  # 针对RegDB数据集
        new_qf = torch.concat([
            getNewFeature(qf, gf, k1, k2),  # 查询-图库关系
            getNewFeature(gf, gf, k1, k2, mean=True),  # 查询-图库关系
        ], dim=1)
        
        new_gf = torch.concat([
            getNewFeature(gf, gf, k1, k2, mean=True),  # 图库内部关系
            getNewFeature(gf, qf, k1, k2)  # 图库-查询关系
        ], dim=1)
    
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