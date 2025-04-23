import torch
import torch.amp as amp
import torch.nn as nn
import torch.distributed as dist
import torch.utils.checkpoint as checkpoint
from  distill_exp.gan.model_cfg import WanModelCFG, sinusoidal_embedding_1d
# from wan.modules.model import WanModel, sinusoidal_embedding_1d
from diffusers.configuration_utils import ConfigMixin, register_to_config
from distill_exp.gan.self_attention import SelfAttention

from fastvideo.utils.parallel_states import get_sequence_parallel_state, nccl_info
from fastvideo.utils.communications import all_gather, all_to_all_4D

class WanExtractor(WanModelCFG):
    r"""
    Wanmin diffusion backbone with only 2 attention blocks, inheriting from Wan model.
    """

    @register_to_config
    def __init__(self,
                model_type='t2v',
                patch_size=(1, 2, 2),
                text_len=512,
                in_dim=16,
                dim=2048,
                ffn_dim=8192,
                freq_dim=256,
                text_dim=4096,
                out_dim=16,
                num_heads=16,
                num_layers=2,  # 只使用2个block
                window_size=(-1, -1),
                qk_norm=True,
                cross_attn_norm=True,
                eps=1e-6,
                with_layer_embedding=True,
                 **kwargs):
        r"""
        Initialize the Wanmin diffusion model backbone with only 2 attention blocks.
        
        Args:
            与基座模型 WanModel 完全相同，但 num_layers 默认为 2
        """
        # 从 WanModelCFG 继承参数和结构
        super().__init__(
            model_type=model_type,
            patch_size=patch_size,
            text_len=text_len,
            in_dim=in_dim,
            dim=dim,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,  # 只使用指定数量的block
            window_size=window_size,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps
        )
        if with_layer_embedding:
            self.layer_embedding = nn.Sequential(
                nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
            )
            for layer in [self.layer_embedding[0],self.layer_embedding[2]]:
                nn.init.normal_(layer.weight, mean=0, std=1e-2)
                nn.init.normal_(layer.bias, mean=0, std=1e-2)
        else:
            self.layer_embedding = None
    
    def load_from_wan(self, wan_model_state_dict):
        """
        从预训练的 WanModel 加载权重
        
        Args:
            wan_model_state_dict: WanModel 的状态字典
            
        Returns:
            missing_keys: 缺失的键列表
            unexpected_keys: 意外的键列表
        """
        # 创建新的state_dict，只保留需要的部分
        new_state_dict = {}
        
        # 复制非blocks部分的权重
        for name, param in wan_model_state_dict.items():
            if not name.startswith('blocks.'):
                new_state_dict[name] = param
            elif int(name.split('.')[1]) < self.config.num_layers:
                # 只复制前 num_layers 个block的权重
                new_state_dict[name] = param
                
        # 加载筛选后的权重
        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
        return missing_keys, unexpected_keys
    
    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        layer_embedding_tensor=None,
        w=104,
        h=60,
        use_checkpoint=True,  # 添加参数控制是否使用checkpoint
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [seq_len, dim]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            use_checkpoint (bool, *optional*):
                Whether to use gradient checkpointing on the transformer blocks

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        with amp.autocast("cuda",dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
        x = [self.head(xi,e) for xi in x]
        gird_tensor = torch.tensor([[21, h//2, w//2]],device=x[0].device,dtype=torch.long)
        x = self.unpatchify(x[0],gird_tensor)

        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        with amp.autocast("cuda",dtype=torch.float32):            
            if self.layer_embedding is not None and layer_embedding_tensor is not None:
                guidance_input = sinusoidal_embedding_1d(self.freq_dim, layer_embedding_tensor).float()
                guidance_emb = self.layer_embedding(guidance_input)
                e = e + guidance_emb

            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)

        print(f"in exractor before block, rank {dist.get_rank()}, x shape: {x.shape}")
        if dist.get_rank()==0:
            print(self.blocks[0])
        for block in self.blocks:
            x = block(x, **kwargs)

        if get_sequence_parallel_state():
            print(f"rank{torch.distributed.get_rank()} in exractor, all_gather x{x.shape}")
            x = all_gather(x,dim=1).contiguous()

        # head
        x = self.head(x, e)

        return x.squeeze()

class QueryTransformerBlock(nn.Module):
    """
    类似 Qformer 的 Transformer 块，用于特征压缩
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, q, kv, attn_mask=None):
        """
        Args:
            q: 查询向量，形状为 [batch_size, q_len, dim]
            kv: 键/值向量，形状为 [batch_size, kv_len, dim]
            attn_mask: 注意力掩码
        """
        # 标准的 Transformer 块设计
        q_norm = self.norm1(q)
        kv_norm = self.norm1(kv)
        attn_output, _ = self.attn(q_norm, kv_norm, kv_norm, attn_mask=attn_mask)
        q = q + attn_output
        q = q + self.mlp(self.norm2(q))
        return q

class FeatureCompressor(nn.Module):
    """
    特征压缩模块，使用类似 Qformer 的结构将特征从 (batch, seq_len, dim) 压缩到 (batch, desired_len, dim)
    """
    def __init__(self, dim, desired_len=16, num_heads=8, mlp_ratio=4.0, dropout=0, num_layer=4):
        super().__init__()
        self.dim = dim
        self.desired_len = desired_len
        
        self.query_tokens = nn.Parameter(torch.zeros(num_layer, desired_len, dim))
        nn.init.normal_(self.query_tokens, std=0.02)
        
        self.block = QueryTransformerBlock(dim, num_heads, mlp_ratio, dropout=dropout)
        
        self.final_norm = nn.LayerNorm(dim)
        
    def forward(self, features):
        """
        Args:
            features: 输入特征，形状为 [batch, seq_len, dim]
            
        Returns:
            compressed_features: 压缩后的特征，形状为 [batch, desired_len, dim]
        """
        query_tokens = self.block(self.query_tokens, features)
            
        compressed_features = self.final_norm(query_tokens)
        
        return compressed_features

class WanGanModel(nn.Module):
    def __init__(self, wan_base, num_layers=2, layer_idxs=[9,19,29,39], dtype=torch.bfloat16, device=None, 
                desired_len=16, compressor_heads=8):
        super().__init__()
        config = wan_base.config
        config = dict(config)
        config["num_layers"] = num_layers
        self.device = device
        self.extractor = WanExtractor(**config).to(device).to(dtype)
        self.layer_tensors = [torch.tensor([layer_idx*100], dtype=torch.float32, device=device) for layer_idx in layer_idxs]

        self.compressor = FeatureCompressor(
            dim=64,
            desired_len=desired_len,
            num_heads=compressor_heads,
            num_layer=len(layer_idxs),
        ).to(device).to(dtype)

        self.self_attn = SelfAttention(dim=64).to(device).to(dtype)

        self.mlp = nn.Sequential(
            nn.LayerNorm(64*desired_len*len(layer_idxs)),
            nn.Linear(64*desired_len*len(layer_idxs), 64),
            nn.GELU(),
            nn.Linear(64, 1)).to(device).to(dtype)

    def forward(self, x, t, context, seq_len,h,w, clip_fea=None, y=None, use_checkpoint=True):
        extracted_features = []
        for x_i, layer_tensor in zip(x, self.layer_tensors):
            output = self.extractor([x_i], t, context, seq_len, layer_embedding_tensor=layer_tensor, clip_fea=clip_fea, y=y, h=h, w=w, use_checkpoint=use_checkpoint)
            extracted_features.append(output)
            
        # 将所有层的输出特征堆叠在一起
        extracted_features = torch.stack(extracted_features, dim=0)  # shape: [num_layers, seq_len, dim]

        # 压缩特征
        compressed_features = self.compressor(extracted_features)  # shape: [num_layers, desired_len, dim]
        compressed_features = compressed_features.flatten(0, 1)  # shape: [num_layers * desired_len, dim]

        # 使用自注意力模块进行进一步处理
        compressed_features = self.self_attn(compressed_features.unsqueeze(0)).squeeze(0) + compressed_features

        compressed_features = compressed_features.flatten(0,1).unsqueeze(0)
        logit = self.mlp(compressed_features)
        
        return logit

def test_extractor():
    """测试 WanExtractor 模型"""
    print("开始测试 WanExtractor 模型...")
    
    # 使用 bf16 精度
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载基座模型
    print("加载基座模型...")
    model_base = WanModel.from_pretrained("/vepfs-zulution/models/Wan2.1-T2V-14B")
    
    # 创建 WanExtractor 模型，使用基座模型的配置
    print("创建 WanExtractor 模型...")
    config = model_base.config
    config = dict(config)
    config["num_layers"] = 2  # 设置为只使用2个block
    print(f"模型配置: {config}")
    extractor = WanExtractor(**config)
    
    # 从基座模型加载权重
    print("从基座模型加载权重...")
    missing_keys, unexpected_keys = extractor.load_from_wan(model_base.state_dict())
    print(f"缺失的键数量: {len(missing_keys)}, 缺失的键: {missing_keys}")
    print(f"意外的键数量: {len(unexpected_keys)}")
    
    # 将模型移动到指定设备并设置为bf16精度
    extractor = extractor.to(device).to(torch.bfloat16)
    extractor.eval()
    
    # 创建模拟输入
    print("创建模拟输入...")
    batch_size = 1
    
    # 创建视频输入（使用bf16精度）
    video = torch.randn(16, 21, 104, 60, device=device, dtype=torch.bfloat16)  # [C, F, H, W]
    x = [video] * batch_size
    
    # 创建时间步
    t = torch.tensor([500] * batch_size, device=device)
    
    # 创建上下文嵌入（使用bf16精度）
    context = [torch.randn(77, 4096, device=device, dtype=torch.bfloat16)] * batch_size
    
    # 设置序列长度
    seq_len = 32760
    
    # 使用 torch.no_grad() 进行推理
    print("开始推理...")
    with torch.no_grad():
        with amp.autocast("cuda",dtype=torch.bfloat16):
            outputs = extractor(x, t, context, seq_len)
        
    print(f"输出特征形状: {outputs.shape}")
    print(f"输出数据类型: {outputs.dtype}")
    print("测试完成!")
    return extractor, outputs

def test_gan_model():
    """测试 WanGanModel 模型"""
    print("\n开始测试 WanGanModel 模型...")
    
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载基座模型
    print("加载基座模型...")
    model_base = WanModel.from_pretrained("/vepfs-zulution/models/Wan2.1-T2V-14B")
    
    # 创建 WanGanModel
    print("创建 WanGanModel...")
    gan_model = WanGanModel(
        model_base, 
        num_layers=2,
        desired_len=16,  # 压缩后的序列长度
        compressor_heads=8,   # 压缩器的注意力头数
        device=device,
        dtype=torch.bfloat16,  # 使用 bf16 精度
    )
    gan_model.train()
    print(f"after set up, memory: {torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024} GB")
    
    num_layer = 4
    bs = 1
    
    # 创建视频输入（使用bf16精度）
    # video = torch.randn(16, 21, 104, 60, device=device, dtype=torch.bfloat16)  # [C, F, H, W]
    video = torch.randn([32760, 5120], device=device, dtype=torch.bfloat16) 
    x = [video] * num_layer
    
    # 创建时间步
    t = torch.tensor([500] * bs, device=device)
    
    # 创建上下文嵌入（使用bf16精度）
    context = [torch.randn(77, 4096, device=device, dtype=torch.bfloat16)] * bs
    
    # 设置序列长度
    seq_len = 32760
    
    with amp.autocast("cuda",dtype=torch.bfloat16):
        outputs = gan_model(x, t, context, seq_len, w=104, h=60,use_checkpoint=True)
    
    print(f"forward cuda max memory: {torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024} GB")
    print(f"GAN 模型输出特征形状: {outputs.shape}")
    print(f"GAN 模型输出数据类型: {outputs.dtype}")

    criterion = nn.BCEWithLogitsLoss()
    fake_labels = torch.zeros_like(outputs, device=device, dtype=torch.bfloat16)
    loss = criterion(outputs, fake_labels)
    loss.backward()
    print(f"GAN 模型损失: {loss.item()}")
    print(f"backward cuda max memory: {torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024} GB")

    print("GAN 模型测试完成!")
    return gan_model, outputs

if __name__ == "__main__":
    # 测试 WanExtractor
    # test_extractor()
    
    # 测试 WanGanModel
    test_gan_model()