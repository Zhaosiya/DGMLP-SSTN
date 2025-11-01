from functools import partial
from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_
from model.block.sstn import SSTN  # 导入SSTN模块

class AttentionGraph(nn.Module):
    """基于注意力的动态图表示模块
    学习关节之间的动态关系，生成动态邻接矩阵
    """
    def __init__(self, in_channels, num_joints, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_joints = num_joints
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        
        # 确保通道数可以被头数整除
        assert self.head_dim * num_heads == in_channels, "in_channels must be divisible by num_heads"
        
        # 查询、键、值投影
        self.q = nn.Linear(in_channels, in_channels)
        self.k = nn.Linear(in_channels, in_channels)
        self.v = nn.Linear(in_channels, in_channels)
        
        # 输出投影
        self.out_proj = nn.Linear(in_channels, in_channels)
        
        # dropout
        self.dropout = nn.Dropout(dropout)
        
        # 温度参数，用于缩放注意力分数
        self.temperature = nn.Parameter(torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)))
        
    def forward(self, x):
        # x: [batch_size, num_joints, in_channels]
        batch_size = x.shape[0]
        
        # 投影到查询、键、值空间
        q = self.q(x)  # [batch_size, num_joints, in_channels]
        k = self.k(x)  # [batch_size, num_joints, in_channels]
        v = self.v(x)  # [batch_size, num_joints, in_channels]
        
        # 重塑以适应多头注意力
        q = rearrange(q, 'b j (h d) -> b h j d', h=self.num_heads)  # [batch_size, num_heads, num_joints, head_dim]
        k = rearrange(k, 'b j (h d) -> b h j d', h=self.num_heads)  # [batch_size, num_heads, num_joints, head_dim]
        v = rearrange(v, 'b j (h d) -> b h j d', h=self.num_heads)  # [batch_size, num_heads, num_joints, head_dim]
        
        # 计算注意力分数
        attn = torch.einsum('bhjd, bhkd -> bhjk', q, k)  # [batch_size, num_heads, num_joints, num_joints]
        attn = attn / self.temperature  # 缩放
        
        # 应用softmax
        attn = F.softmax(attn, dim=-1)  # [batch_size, num_heads, num_joints, num_joints]
        attn = self.dropout(attn)
        
        # 应用注意力到值
        out = torch.einsum('bhjk, bhkd -> bhjd', attn, v)  # [batch_size, num_heads, num_joints, head_dim]
        
        # 重塑回原始形状
        out = rearrange(out, 'b h j d -> b j (h d)')  # [batch_size, num_joints, in_channels]
        
        # 输出投影
        out = self.out_proj(out)  # [batch_size, num_joints, in_channels]
        
        # 添加残差连接
        out = out + x
        
        return out, attn

class Gcn(nn.Module):
    def __init__(self, in_channels, out_channels, adj):
        super().__init__()
        self.adj = adj
        self.kernel_size = adj.size(0)
        self.conv = nn.Conv2d(in_channels, out_channels * self.kernel_size, kernel_size=(1, 1))

    def forward(self, x):
        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv, kvw->nctw', (x, self.adj))

        return x.contiguous()


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class Mlp_ln(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features)
        )

        self.act = act_layer()

        self.fc2 = nn.Sequential(
            nn.Linear(hidden_features, out_features),
            nn.LayerNorm(out_features)
        )

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class Block(nn.Module):
    def __init__(self, length, frames, dim, tokens_dim, channels_dim, adj, drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(length)

        # 在GCN之前添加SSTN模块
        #self.sstn = SSTN(dim)
        # 添加注意力模块
        self.attention = AttentionGraph(dim, length)
        self.gcn_1 = Gcn(dim, dim, adj)
        self.gcn_2 = Gcn(dim, dim, adj)
        self.adj = adj

        if frames == 1:
            self.mlp_1 = Mlp(in_features=length, hidden_features=tokens_dim, act_layer=act_layer, drop=drop)
        else:
            self.mlp_1 = Mlp_ln(in_features=length, hidden_features=tokens_dim, act_layer=act_layer, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_2 = Mlp(in_features=dim, hidden_features=channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        ## Spatial Graph MLP
        x = rearrange(x, f'b j c -> b j c')
        res = x
        x = self.norm1(x.transpose(1, 2)).transpose(1, 2)

        # 应用SSTN空间变换
        #x = self.sstn(x)

        # 应用注意力机制
        x, attn = self.attention(x)

        x_gcn_1 = rearrange(x, 'b j c-> b c 1 j')
        x_gcn_1 = self.gcn_1(x_gcn_1)
        x_gcn_1 = rearrange(x_gcn_1, 'b c 1 j -> b j c')

        x = res + self.drop_path(self.mlp_1(x.transpose(1, 2)).transpose(1, 2) + x_gcn_1)

        ## Channel Graph MLP
        x = rearrange(x, f'b j c -> b j c')
        res = x
        x = self.norm2(x)

        # 移除第二个SSTN应用

        x_gcn_2 = rearrange(x, 'b j c-> b c 1 j')
        x_gcn_2 = self.gcn_2(x_gcn_2)
        x_gcn_2 = rearrange(x_gcn_2, 'b c 1 j -> b j c')

        x = res + self.drop_path(self.mlp_2(x) + x_gcn_2)

        return x


class Mlp_gcn(nn.Module):
    def __init__(self, depth, embed_dim, channels_dim, tokens_dim, adj, drop_rate=0.10, length=17, frames=1):
        super().__init__()
        drop_path_rate = 0.2

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            Block(
                length, frames, embed_dim, tokens_dim, channels_dim, adj,
                drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x