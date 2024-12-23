import torch
from torch import nn
from torch.nn import init

# 定义ECA注意力模块的类
class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)  # 定义全局平均池化层，将空间维度压缩为1x1
        # 定义一个1D卷积，用于处理通道间的关系，核大小可调，padding保证输出通道数不变
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数，用于激活最终的注意力权重

    # 权重初始化方法
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')  # 对Conv2d层使用Kaiming初始化
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 如果有偏置项，则初始化为0
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)  # 批归一化层权重初始化为1
                init.constant_(m.bias, 0)  # 批归一化层偏置初始化为0
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)  # 全连接层权重使用正态分布初始化
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 全连接层偏置初始化为0

    # 前向传播方法
    def forward(self, x):
        y = self.gap(x)  # 对输入x应用全局平均池化，得到bs,c,1,1维度的输出
        y = y.squeeze(-1).permute(0, 2, 1)  # 移除最后一个维度并转置，为1D卷积准备，变为bs,1,c
        y = self.conv(y)  # 对转置后的y应用1D卷积，得到bs,1,c维度的输出
        y = self.sigmoid(y)  # 应用Sigmoid函数激活，得到最终的注意力权重
        y = y.permute(0, 2, 1).unsqueeze(-1)  # 再次转置并增加一个维度，以匹配原始输入x的维度
        return x * y.expand_as(x)  # 将注意力权重应用到原始输入x上，通过广播机制扩展维度并执行逐元素乘法



class SEAttention(nn.Module):
    # 初始化SE模块，channel为通道数，reduction为降维比率
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化层，将特征图的空间维度压缩为1x1
        self.fc = nn.Sequential(  # 定义两个全连接层作为激励操作，通过降维和升维调整通道重要性
            nn.Linear(channel, channel // reduction, bias=False),  # 降维，减少参数数量和计算量
            nn.ReLU(inplace=True),  # ReLU激活函数，引入非线性
            nn.Linear(channel // reduction, channel, bias=False),  # 升维，恢复到原始通道数
            nn.Sigmoid()  # Sigmoid激活函数，输出每个通道的重要性系数
        )

    # 权重初始化方法
    def init_weights(self):
        for m in self.modules():  # 遍历模块中的所有子模块
            if isinstance(m, nn.Conv2d):  # 对于卷积层
                init.kaiming_normal_(m.weight, mode='fan_out')  # 使用Kaiming初始化方法初始化权重
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 如果有偏置项，则初始化为0
            elif isinstance(m, nn.BatchNorm2d):  # 对于批归一化层
                init.constant_(m.weight, 1)  # 权重初始化为1
                init.constant_(m.bias, 0)  # 偏置初始化为0
            elif isinstance(m, nn.Linear):  # 对于全连接层
                init.normal_(m.weight, std=0.001)  # 权重使用正态分布初始化
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 偏置初始化为0

    # 前向传播方法
    def forward(self, x):
        b, c, _, _ = x.size()  # 获取输入x的批量大小b和通道数c
        y = self.avg_pool(x).view(b, c)  # 通过自适应平均池化层后，调整形状以匹配全连接层的输入
        y = self.fc(y).view(b, c, 1, 1)  # 通过全连接层计算通道重要性，调整形状以匹配原始特征图的形状
        return x * y.expand_as(x)  # 将通道重要性系数应用到原始特征图上，进行特征重新校准

# # 示例使用
# if __name__ == '__main__':
#     block = ECAAttention(kernel_size=3)  # 实例化ECA注意力模块，指定核大小为3
#     input = torch.rand(1, 64, 64, 64)  # 生成一个随机输入
#     output = block(input)  # 将输入通过ECA模块处理
#     print(input.size(), output.size())  # 打印输入和输出的尺寸，验证ECA模块的作用
# 导入必要的PyTorch模块
import torch
from torch import nn
from torch.nn import functional as F

class CoTAttention(nn.Module):
    # 初始化CoT注意力模块
    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.dim = dim  # 输入的通道数
        self.kernel_size = kernel_size  # 卷积核大小

        # 定义用于键(key)的卷积层，包括一个分组卷积，BatchNorm和ReLU激活
        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=4, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )

        # 定义用于值(value)的卷积层，包括一个1x1卷积和BatchNorm
        self.value_embed = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

        # 缩小因子，用于降低注意力嵌入的维度
        factor = 4
        # 定义注意力嵌入层，由两个卷积层、一个BatchNorm层和ReLU激活组成
        self.attention_embed = nn.Sequential(
            nn.Conv2d(2*dim, 2*dim//factor, 1, bias=False),
            nn.BatchNorm2d(2*dim//factor),
            nn.ReLU(),
            nn.Conv2d(2*dim//factor, kernel_size*kernel_size*dim, 1)
        )

    def forward(self, x):
        # 前向传播函数
        bs, c, h, w = x.shape  # 输入特征的尺寸
        k1 = self.key_embed(x)  # 生成键的静态表示
        v = self.value_embed(x).view(bs, c, -1)  # 生成值的表示并调整形状

        y = torch.cat([k1, x], dim=1)  # 将键的静态表示和原始输入连接
        att = self.attention_embed(y)  # 生成动态注意力权重
        att = att.reshape(bs, c, self.kernel_size*self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)  # 计算注意力权重的均值并调整形状
        k2 = F.softmax(att, dim=-1) * v  # 应用注意力权重到值上
        k2 = k2.view(bs, c, h, w)  # 调整形状以匹配输出

        return k1 + k2  # 返回键的静态和动态表示的总和

# # 实例化CoTAttention模块并测试
# if __name__ == '__main__':
#     block = CoTAttention(64)  # 创建一个输入通道数为64的CoTAttention实例
#     input = torch.rand(1, 64, 64, 64)  # 创建一个随机输入
#     output = block(input)  # 通过CoTAttention模块处理输入
#     print(output.shape)  # 打印输入和输出的尺寸


import torch
from torch import nn
import torch.nn.functional as F

# 用于调整数值，使其可以被某个除数整除，常用于网络层中通道数的设置。
def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # 确保减小的百分比不超过一定的比例（round_limit）
    if new_v < round_limit * v:
        new_v += divisor
    return new_v

# Radix Softmax用于处理分组特征的归一化
class RadixSoftmax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        # 根据radix是否大于1来决定使用softmax还是sigmoid进行归一化
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = x.sigmoid()
        return x

# SplitAttn模块实现分裂注意力机制
class SplitAttn(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1, padding=None,
                 dilation=1, groups=1, bias=False, radix=2, rd_ratio=0.25, rd_channels=None, rd_divisor=8,
                 act_layer=nn.ReLU, norm_layer=None, drop_block=None, **kwargs):
        super(SplitAttn, self).__init__()
        out_channels = out_channels or in_channels
        self.radix = radix
        self.drop_block = drop_block
        mid_chs = out_channels * radix
        # 根据输入通道数、radix和rd_ratio计算注意力机制的中间层通道数
        if rd_channels is None:
            attn_chs = make_divisible(
                in_channels * radix * rd_ratio, min_value=32, divisor=rd_divisor)
        else:
            attn_chs = rd_channels * radix

        padding = kernel_size // 2 if padding is None else padding
        # 核心卷积层
        self.conv = nn.Conv2d(
            in_channels, mid_chs, kernel_size, stride, padding, dilation,
            groups=groups * radix, bias=bias, **kwargs)
        # 后续层以及RadixSoftmax
        self.bn0 = norm_layer(mid_chs) if norm_layer else nn.Identity()
        self.act0 = act_layer()
        self.fc1 = nn.Conv2d(out_channels, attn_chs, 1, groups=groups)
        self.bn1 = norm_layer(attn_chs) if norm_layer else nn.Identity()
        self.act1 = act_layer()
        self.fc2 = nn.Conv2d(attn_chs, mid_chs, 1, groups=groups)
        self.rsoftmax = RadixSoftmax(radix, groups)

    def forward(self, x):
        # 卷积和激活
        x = self.conv(x)
        x = self.bn0(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act0(x)

        # 计算分裂注意力
        B, RC, H, W = x.shape
        if self.radix > 1:
            # 对特征进行重组和聚合
            x = x.reshape((B, self.radix, RC // self.radix, H, W))
            x_gap = x.sum(dim=1)
        else:
            x_gap = x
        # 全局平均池化和两层全连接网络，应用RadixSoftmax
        x_gap = x_gap.mean(2, keepdims=True).mean(3, keepdims=True)
        x_gap = self.fc1(x_gap)
        x_gap = self.bn1(x_gap)
        x_gap = self.act1(x_gap)
        x_attn = self.fc2(x_gap)
        x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        if self.radix > 1:
            out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        else:
            out = x * x_attn
        return out



# # 输入 N C H W,  输出 N C H W
# if __name__ == '__main__':
#     block = SplitAttn(64)
#     input = torch.rand(1, 64, 64, 64)
#     output = block(input)
#     print(output.shape)
import numpy as np
import torch
from torch import nn
from torch.nn import init

def spatial_shift1(x):
    # 实现第一种空间位移，位移图像的四分之一块
    b, w, h, c = x.size()
    # 以下四行代码分别向左、向右、向上、向下移动图像的四分之一块
    x[:, 1:, :, :c // 4] = x[:, :w - 1, :, :c // 4]
    x[:, :w - 1, :, c // 4:c // 2] = x[:, 1:, :, c // 4:c // 2]
    x[:, :, 1:, c // 2:c * 3 // 4] = x[:, :, :h - 1, c // 2:c * 3 // 4]
    x[:, :, :h - 1, 3 * c // 4:] = x[:, :, 1:, 3 * c // 4:]
    return x

def spatial_shift2(x):
    # 实现第二种空间位移，逻辑与spatial_shift1相似，但位移方向不同
    b, w, h, c = x.size()
    # 对图像的四分之一块进行空间位移
    x[:, :, 1:, :c // 4] = x[:, :, :h - 1, :c // 4]
    x[:, :, :h - 1, c // 4:c // 2] = x[:, :, 1:, c // 4:c // 2]
    x[:, 1:, :, c // 2:c * 3 // 4] = x[:, :w - 1, :, c // 2:c * 3 // 4]
    x[:, :w - 1, :, 3 * c // 4:] = x[:, 1:, :, 3 * c // 4:]
    return x

class SplitAttention(nn.Module):
    # 定义分割注意力模块，使用MLP层进行特征转换和注意力权重计算
    def __init__(self, channel=2048, k=3):
        super().__init__()
        self.channel = channel
        self.k = k  # 分割的块数
        # 定义MLP层和激活函数
        self.mlp1 = nn.Linear(channel, channel, bias=False)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(channel, channel * k, bias=False)
        self.softmax = nn.Softmax(1)

    def forward(self, x_all):
        # 计算分割注意力，并应用于输入特征
        b, k, h, w, c = x_all.shape
        x_all = x_all.reshape(b, k, -1, c)  # 重塑维度
        a = torch.sum(torch.sum(x_all, 1), 1)  # 聚合特征
        hat_a = self.mlp2(self.gelu(self.mlp1(a)))  # 通过MLP计算注意力权重
        hat_a = hat_a.reshape(b, self.k, c)  # 调整形状
        bar_a = self.softmax(hat_a)  # 应用softmax获取注意力分布
        attention = bar_a.unsqueeze(-2)  # 增加维度
        out = attention * x_all  # 将注意力权重应用于特征
        out = torch.sum(out, 1).reshape(b, h, w, c)  # 聚合并调整形状
        return out

class S2Attention(nn.Module):
    # S2注意力模块，整合空间位移和分割注意力
    def __init__(self, channels=2048):
        super().__init__()
        # 定义MLP层
        self.mlp1 = nn.Linear(channels, channels * 3)
        self.mlp2 = nn.Linear(channels, channels)
        self.split_attention = SplitAttention()

    def forward(self, x):
        b, c, w, h = x.size()
        x = x.permute(0, 2, 3, 1)  # 调整维度顺序
        x = self.mlp1(x)  # 通过MLP层扩展特征
        x1 = spatial_shift1(x[:, :, :, :c])  # 应用第一种空间位移
        x2 = spatial_shift2(x[:, :, :, c:c * 2])  # 应用第二种空间位移
        x3 = x[:, :, :, c * 2:]  # 保留原始特征的一部分
        x_all = torch.stack([x1, x2, x3], 1)  # 堆叠特征
        a = self.split_attention(x_all)  # 应用分割注意力
        x = self.mlp2(a)  # 通过另一个MLP层缩减特征维度
        x = x.permute(0, 3, 1, 2)  # 调整维度顺序回原始
        return x

# # 示例代码
# if __name__ == '__main__':
#     input = torch.randn(50, 512, 7, 7)  # 创建输入张量
#     s2att = S2Attention(channels=512)  # 实例化S2注意力模块
#     output = s2att(input)  # 通过S2注意力模块处理输入
#     print(output.shape)  # 打印输出张量的形状
class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 使用5x5核的卷积层，应用深度卷积
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        # 两组卷积层，分别使用1x7和7x1核，用于跨度不同的特征提取，均应用深度卷积
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        # 另外两组卷积层，使用更大的核进行特征提取，分别为1x11和11x1，也是深度卷积
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        # 使用最大尺寸的核进行特征提取，为1x21和21x1，深度卷积
        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        # 最后一个1x1卷积层，用于整合上述所有特征提取的结果
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone() # 克隆输入x，以便之后与注意力加权的特征进行相乘
        attn = self.conv0(x) # 应用初始的5x5卷积

        # 应用1x7和7x1卷积，进一步提取特征
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        # 应用1x11和11x1卷积，进一步提取特征
        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        # 应用1x21和21x1卷积，进一步提取特征
        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2 # 将所有特征提取的结果相加

        attn = self.conv3(attn) # 应用最后的1x1卷积层整合特征

        return attn * u # 将原始输入和注意力加权的特征相乘，返回最终结果
    
import torch
from torch import nn

# 定义EMA模块
class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        # 设置分组数量，用于特征分组
        self.groups = factor
        # 确保分组后的通道数大于0
        assert channels // self.groups > 0
        # softmax激活函数，用于归一化
        self.softmax = nn.Softmax(-1)
        # 全局平均池化，生成通道描述符
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        # 水平方向的平均池化，用于编码水平方向的全局信息
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        # 垂直方向的平均池化，用于编码垂直方向的全局信息
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        # GroupNorm归一化，减少内部协变量偏移
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        # 1x1卷积，用于学习跨通道的特征
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        # 3x3卷积，用于捕捉更丰富的空间信息
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        # 对输入特征图进行分组处理
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        # 应用水平和垂直方向的全局平均池化
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        # 通过1x1卷积和sigmoid激活函数，获得注意力权重
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        # 应用GroupNorm和注意力权重调整特征图
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        # 将特征图通过全局平均池化和softmax进行处理，得到权重
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        # 通过矩阵乘法和sigmoid激活获得最终的注意力权重，调整特征图
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        # 将调整后的特征图重塑回原始尺寸
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)



    
import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticPartAttention(nn.Module):
    def __init__(self, in_channels = 2048, part_num=7, reduction_ratio=16):
        super(SemanticPartAttention, self).__init__()
        self.part_num = part_num
        
        # Part-wise Embedding
        self.part_embedding = nn.Parameter(torch.randn(part_num, in_channels))
        
        # Channel Attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Spatial Attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # Part Diversity Loss Computation
        self.part_diversity_loss_fn = nn.MSELoss()
    
    def forward(self, x):
        B, C, H, W = x.shape

        # Channel Attention
        channel_attn = self.channel_attention(x)
        x_channel_attn = x * channel_attn

        # Spatial Attention
        avg_out = torch.mean(x_channel_attn, dim=1, keepdim=True)
        max_out, _ = torch.max(x_channel_attn, dim=1, keepdim=True)
        spatial_attn = torch.cat([avg_out, max_out], dim=1)
        spatial_attn = self.spatial_attention(spatial_attn)
        x_spatial_attn = x_channel_attn * spatial_attn

        # Part-wise Attention
        x_reshaped = x_spatial_attn.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        part_attn = F.softmax(torch.matmul(x_reshaped, self.part_embedding.T), dim=-1)  # [B, H*W, part_num]

        # 将part特征展平为二维
        part_features = torch.matmul(part_attn.transpose(1, 2), x_reshaped)  # [B, part_num, C]
        part_features = part_features.reshape(B, -1)  # 展平为 [B, part_num * C]

        # Compute Part Diversity Loss
        if self.training:
            part_diversity_loss = self._compute_part_diversity_loss(part_attn)
        else:
            part_diversity_loss = None

        return part_features, part_attn, part_diversity_loss

    def _compute_part_diversity_loss(self, part_attn):
        """
        Compute part diversity loss to encourage different parts to focus on different regions
        """
        diversity_loss = 0
        for i in range(self.part_num):
            for j in range(i+1, self.part_num):
                diversity_loss += torch.mean(part_attn[:, i] * part_attn[:, j])
        
        return diversity_loss / (self.part_num * (self.part_num - 1) / 2)
    