import torch
import torch.nn as nn

# ====================== 网络结构 ======================
class EnhancedAttentionNetwork(nn.Module):
    """增强型注意力网络，用于DQN的策略和目标网络

    特性：
    - 增强的多头注意力机制：以望远镜状态为 query,对全部目标特征做注意力
    - 残差连接 + Layer Normalization
    - 每个目标在输入中占据固定位置,输出第 i 维 Q 值恒定对应目标 i
    - dueling=True 时采用决斗结构: Q = V + (A - mean(A)),
      把状态价值与动作优势解耦,多目标编排中大量动作等价时收敛更快
    """
    def __init__(self, input_dim: int, action_dim: int, embedding_dim: int = 128,
                 num_heads: int = 8, obs_feat_dim: int = 4, target_feat_dim: int = 6,
                 dueling: bool = True):
        super().__init__()
        self.obs_feat_dim = obs_feat_dim
        self.target_feat_dim = target_feat_dim
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.num_targets = (input_dim - obs_feat_dim) // target_feat_dim
        self.dueling = dueling

        # 观测特征编码器(望远镜全局状态)
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_feat_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim)
        )

        # 目标特征编码器
        self.target_encoder = nn.Sequential(
            nn.Linear(target_feat_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim)
        )

        # 增强的多头注意力层
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=0.1)

        # 后处理层 - 残差连接和层归一化
        self.layer_norm = nn.LayerNorm(embedding_dim)

        if dueling:
            # 决斗结构: 状态价值流 + 动作优势流
            self.value_stream = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, 1)
            )
            self.advantage_stream = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(embedding_dim * 2, action_dim)
            )
        else:
            self.output_layers = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, action_dim)
            )

        # 初始化网络权重
        self.init_weights()

    def init_weights(self):
        """Xavier/Glorot 初始化,提高训练稳定性和收敛速度"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        if hasattr(self.attention, 'in_proj_weight') and self.attention.in_proj_weight is not None:
            nn.init.xavier_uniform_(self.attention.in_proj_weight)
        if hasattr(self.attention, 'out_proj'):
            nn.init.xavier_uniform_(self.attention.out_proj.weight)
            nn.init.zeros_(self.attention.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播: x = [全局特征(obs_feat_dim) | 目标0特征 | 目标1特征 | ...]"""
        obs_features = x[:, :self.obs_feat_dim]
        n = self.num_targets * self.target_feat_dim
        target_features = x[:, self.obs_feat_dim:self.obs_feat_dim + n] \
            .view(x.size(0), self.num_targets, self.target_feat_dim)

        # 编码观测状态
        obs_emb = self.obs_encoder(obs_features).unsqueeze(1)  # [B, 1, E]

        # 编码目标特征
        target_emb = self.target_encoder(target_features)      # [B, T, E]

        # 注意力机制
        query = obs_emb.transpose(0, 1)    # [1, B, E]
        key = target_emb.transpose(0, 1)   # [T, B, E]
        value = target_emb.transpose(0, 1) # [T, B, E]

        attn_output, _ = self.attention(query, key, value)

        # 残差连接和层归一化
        attn_output = attn_output + query
        attn_output = self.layer_norm(attn_output)
        h = attn_output.squeeze(0)  # [B, E]

        if not self.dueling:
            return self.output_layers(h)  # [B, action_dim]

        # 决斗组合: Q(s,a) = V(s) + A(s,a) - mean_a' A(s,a')
        v = self.value_stream(h)                       # [B, 1]
        a = self.advantage_stream(h)                   # [B, action_dim]
        return v + (a - a.mean(dim=1, keepdim=True))
