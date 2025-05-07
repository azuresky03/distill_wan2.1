import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    """
    标准的 Self-Attention 模块，接收形状为 [bs, seq_len, dim] 的输入
    
    参数:
        dim (int): 输入特征维度
        num_heads (int): 注意力头数量
        dropout (float): dropout 比率
        use_flash_attn (bool): 是否使用 Flash Attention (如果可用)
        qk_norm (bool): 是否对 query 和 key 进行归一化
    """
    
    def __init__(
        self,
        dim,
        num_heads=8,
        dropout=0.0,
        use_flash_attn=True,
        qk_norm=True,
        scale_factor=None,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"特征维度 {dim} 必须能被头数 {num_heads} 整除"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = scale_factor or (self.head_dim ** -0.5)
        self.use_flash_attn = use_flash_attn
        self.qk_norm = qk_norm
        
        # 定义线性投影层
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # 如果需要对 q 和 k 进行归一化
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def reshape_for_attention(self, x):
        """将输入张量重塑为注意力计算所需的形状"""
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    
    def forward(self, x, attn_mask=None):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入张量，形状为 [batch_size, seq_len, dim]
            attn_mask (torch.Tensor, 可选): 注意力掩码，形状为 [batch_size, seq_len, seq_len]
                                           或 [batch_size, 1, seq_len, seq_len]
        
        返回:
            torch.Tensor: 自注意力处理后的输出，形状为 [batch_size, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # 线性投影
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 重塑为多头形式
        q = self.reshape_for_attention(q)  # [bs, num_heads, seq_len, head_dim]
        k = self.reshape_for_attention(k)  # [bs, num_heads, seq_len, head_dim]
        v = self.reshape_for_attention(v)  # [bs, num_heads, seq_len, head_dim]
        
        # 对 query 和 key 进行归一化（如果需要）
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        # 计算注意力
        if self.use_flash_attn and hasattr(F, 'scaled_dot_product_attention'):
            # 使用更快的 PyTorch 2.0 注意力实现
            q = q.transpose(1, 2)  # [bs, seq_len, num_heads, head_dim]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            attn_output = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )
            
            # 转回原来的形状
            attn_output = attn_output.transpose(1, 2)  # [bs, num_heads, seq_len, head_dim]
        else:
            # 标准注意力计算
            attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [bs, num_heads, seq_len, seq_len]
            
            # 应用注意力掩码（如果提供）
            if attn_mask is not None:
                if attn_mask.dim() == 3:
                    attn_mask = attn_mask.unsqueeze(1)  # [bs, 1, seq_len, seq_len]
                attn_weights = attn_weights.masked_fill(attn_mask == 0, float('-inf'))
                
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # 应用注意力权重
            attn_output = torch.matmul(attn_weights, v)  # [bs, num_heads, seq_len, head_dim]
        
        # 重组头部并投影回输出空间
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        output = self.out_proj(attn_output)
        
        return output


class TransformerBlock(nn.Module):
    """
    标准的 Transformer Block，包含 self-attention 和前馈网络
    
    参数:
        dim (int): 输入特征维度
        num_heads (int): 注意力头数量
        mlp_ratio (float): MLP 中间层维度与输入维度的比率
        dropout (float): dropout 比率
        use_flash_attn (bool): 是否使用 Flash Attention (如果可用)
        qk_norm (bool): 是否对 query 和 key 进行归一化
    """
    
    def __init__(
        self,
        dim,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
        use_flash_attn=False,
        qk_norm=False,
    ):
        super().__init__()
        self.dim = dim
        
        # Self-attention
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = SelfAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            use_flash_attn=use_flash_attn,
            qk_norm=qk_norm,
        )
        
        # Feed-forward network
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )
        
    def forward(self, x, attn_mask=None):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入张量，形状为 [batch_size, seq_len, dim]
            attn_mask (torch.Tensor, 可选): 注意力掩码
        
        返回:
            torch.Tensor: Transformer 块处理后的输出，形状为 [batch_size, seq_len, dim]
        """
        # Self-attention 部分 (残差连接 + LayerNorm)
        x = x + self.self_attn(self.norm1(x), attn_mask=attn_mask)
        
        # Feed-forward 部分 (残差连接 + LayerNorm)
        x = x + self.mlp(self.norm2(x))
        
        return x


# 使用示例
if __name__ == "__main__":
    # 创建自注意力模块
    batch_size, seq_len, dim = 2, 128, 512
    num_heads = 8
    
    # 创建输入张量
    x = torch.randn(batch_size, seq_len, dim)
    
    # 测试独立的 SelfAttention
    self_attn = SelfAttention(dim=dim, num_heads=num_heads)
    output = self_attn(x)
    print(f"Self-Attention 输出形状: {output.shape}")
    
    # 测试整个 TransformerBlock
    transformer_block = TransformerBlock(dim=dim, num_heads=num_heads)
    output = transformer_block(x)
    print(f"Transformer Block 输出形状: {output.shape}")