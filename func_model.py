import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import os
os.environ['USE_PYGEOS'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ====================== 核心因果推断组件 ======================
class DagLayer(nn.Module):
    """有向无环图层 - 支持结构干预和原始矩阵保留"""
    def __init__(self, in_features, out_features, bias=False, initial=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 初始化邻接矩阵
        self.a = torch.zeros(out_features, out_features)
        if initial:
            # 初始因果结构：每个变量影响后续变量
            for i in range(out_features):
                for j in range(i+1, out_features):
                    self.a[i][j] = 1.0
        
        self.A = nn.Parameter(self.a)
        self.I = torch.eye(out_features, device=device)
        self.I.requires_grad = False
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, z):
        """应用DAG变换到输入数据"""
        # 使用tanh激活确保值在合理范围
        A_processed = torch.tanh(self.A) * (1 - self.I)
        # 应用因果变换：c = A^T c + u
        return torch.matmul(A_processed.T, z.unsqueeze(-1)).squeeze(-1)
    
    def get_adjacency_matrix(self, threshold=0.1):
        """获取当前邻接矩阵（用于因果分析）"""
        A_val = torch.tanh(self.A) * (1 - self.I)
        return (torch.abs(A_val) > threshold).float().detach().cpu().numpy()
    
    def get_raw_adjacency_matrix(self):
        """获取原始未二值化的邻接矩阵（用于详细分析）"""
        return torch.tanh(self.A).detach().cpu().numpy()
    
    def notears_constraint(self):
        """NOTEARS无环约束"""
        A_square = self.A * self.A
        matrix_exp = torch.matrix_exp(A_square)
        trace = torch.trace(matrix_exp)
        return trace - self.out_features
    
    def apply_intervention(self, source_idx, target_idx, intervention_type='remove', strength=0.0):
        """
        应用结构干预到邻接矩阵
        参数:
            source_idx: 源变量索引
            target_idx: 目标变量索引
            intervention_type: 干预类型 ('remove', 'strengthen', 'weaken', 'set')
            strength: 干预强度 (0.0-1.0)
        """
        # 边界检查
        assert 0 <= source_idx < self.out_features, f"无效来源索引: {source_idx}"
        assert 0 <= target_idx < self.out_features, f"无效目标索引: {target_idx}"
        
        # 获取当前权重值 (经过tanh处理后的实际权重)
        current_weight = torch.tanh(self.A.data[source_idx, target_idx]).item()
        
        if intervention_type == 'remove':
            # 完全移除关系
            self.A.data[source_idx, target_idx] = 0.0
            
        elif intervention_type == 'strengthen':
            # 强化关系 (增加权重)
            new_weight = min(0.999, current_weight + strength * (1 - current_weight))
            self.A.data[source_idx, target_idx] = torch.tensor(np.arctanh(new_weight))
            
        elif intervention_type == 'weaken':
            # 弱化关系 (减少权重)
            new_weight = max(0.001, current_weight - strength * current_weight)
            self.A.data[source_idx, target_idx] = torch.tensor(np.arctanh(new_weight))
            
        elif intervention_type == 'set':
            # 直接设置权重值
            assert 0 <= strength <= 1, "权重值必须在0-1之间"
            self.A.data[source_idx, target_idx] = torch.tensor(np.arctanh(strength))
            
        else:
            raise ValueError(f"未知干预类型: {intervention_type}")
        
        # 打印干预日志
        new_val = torch.tanh(self.A.data[source_idx, target_idx]).item()
        print(f"干预: {source_idx}→{target_idx} | "
              f"类型: {intervention_type} | "
              f"强度: {strength} | "
              f"权重: {current_weight:.4f} → {new_val:.4f}")

class SimplifiedMask(nn.Module):
    """简化的潜变量掩码层 - 用于因果干预分析"""
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 32),
            nn.ReLU(),
            nn.Linear(32, z_dim)
        )
    
    def forward(self, z):
        """应用掩码变换"""
        return self.net(z)

class SpatialAwareTabularEncoder(nn.Module):
    """包含空间滞后特征的表格数据编码器"""
    def __init__(self, input_dim, z_dim):
        """
        input_dim: 输入特征维度（城市变量数量）
        z_dim: 潜变量维度
        """
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        
        # 编码器网络 - 输入维度增加空间滞后特征
        self.net = nn.Sequential(
            nn.Linear(2 * input_dim, 32),  # 原始特征 + 空间滞后特征
            nn.ELU(),
            nn.Linear(32, 16),
            nn.ELU(),
            nn.Linear(16, 2 * z_dim)  # 输出均值和方差
        )
    
    def forward(self, x, spatial_lags):
        """编码输入数据为潜变量分布"""
        # 拼接原始特征和空间滞后特征
        combined = torch.cat([x, spatial_lags], dim=1)
        h = self.net(combined)
        
        # 分割为均值和方差
        m, v = h.chunk(2, dim=-1)
        # 确保方差为正
        v = F.softplus(v) + 1e-8
        return m, v

class TabularDecoderDAG(nn.Module):
    """带DAG结构的表格数据解码器 - 用于城市能耗预测"""
    def __init__(self, z_dim, output_dim):
        """
        z_dim: 潜变量维度
        output_dim: 输出维度（城市变量数量）
        """
        super().__init__()
        self.z_dim = z_dim
        self.output_dim = output_dim
        
        # 解码器网络
        self.net = nn.Sequential(
            nn.Linear(z_dim, 16),
            nn.ELU(),
            nn.Linear(16, 32),
            nn.ELU(),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, z):
        """从潜变量重建原始特征"""
        return self.net(z)

def kl_normal(qm, qv, pm, pv):
    """
    Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
    sum over the last dimension
    Args:
        qm: tensor: (batch, dim): q mean
        qv: tensor: (batch, dim): q variance
        pm: tensor: (batch, dim): p mean
        pv: tensor: (batch, dim): p variance
    Return:
        kl: tensor: (batch,): kl between each sample
    """
    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
    kl = element_wise.sum(-1)
    return kl

class SpatialUrbanCausalVAE(nn.Module):
    def __init__(self, input_dim, spatial_weight_matrix, lisa_cluster, coords, z1_dim=11, z2_dim=1, 
                 spatial_reg_weight=0.1, lag_reg_weight=0.01, dag_weight=0.1):
        """
        参数:
            input_dim: 输入特征维度 (11: 10个特征 + 1个目标变量)
            spatial_weight_matrix: 空间权重矩阵 (PyTorch张量)
            lisa_cluster: LISA聚类结果 (numpy数组)
            z1_dim: 因果因子数量 (与现有VAE的c_dim一致)
            z2_dim: 每个因果因子的维度 (设为1，标量)
            spatial_reg_weight: 空间正则化权重
            lag_reg_weight: 空间滞后正则化权重
        """
        super().__init__()
        self.input_dim = input_dim
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.z_dim = z1_dim * z2_dim  # 总潜变量维度
        self.spatial_reg_weight = spatial_reg_weight
        self.lag_reg_weight = lag_reg_weight
        self.dag_weight = dag_weight
        self.register_buffer('coords', coords)
        
        # 存储空间关系数据
        self.register_buffer('spatial_weight', spatial_weight_matrix)
        self.register_buffer('lisa_cluster', torch.tensor(lisa_cluster, dtype=torch.long))
        
        # 编码器 - 包含空间滞后特征
        self.enc = SpatialAwareTabularEncoder(input_dim, self.z_dim)
        
        # 解码器
        self.dec = TabularDecoderDAG(self.z_dim, input_dim)
        
        # DAG层
        self.dag = DagLayer(self.z1_dim, self.z1_dim)
        
        # 潜变量掩码层
        self.mask_z = SimplifiedMask(self.z_dim)
        
        # 空间聚类嵌入层 (用于HH/LL聚类正则化)
        self.cluster_embedding = nn.Embedding(5, self.z_dim)  # 0:不显著, 1:HH, 2:LH, 3:LL, 4:HL
    
    def encode(self, x, spatial_lags):
        """编码输入数据，包含空间滞后特征"""
        return self.enc(x, spatial_lags)
    
    def reparameterize(self, mu, var):
        """重参数化技巧"""
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def apply_dag(self, z):
        """应用因果结构到潜变量"""
        # 重塑为因果因子形式 [batch_size, z1_dim, z2_dim]
        z_reshaped = z.view(-1, self.z1_dim, self.z2_dim)
        
        # 应用DAG变换
        z_dag = self.dag(z_reshaped.squeeze(-1))  # 移除z2_dim维度
        
        return z_dag
    
    def decode(self, z):
        """从潜变量重建数据"""
        return self.dec(z)
    
    def forward(self, x, spatial_lags):
        """前向传播 - 包含空间滞后特征"""
        # 编码 (包含空间滞后特征)
        q_m, q_v = self.encode(x, spatial_lags)
        z = self.reparameterize(q_m, q_v)
        
        # 应用因果结构
        z_dag = self.apply_dag(z)
        
        # 应用掩码
        z_masked = self.mask_z(z_dag.view(-1, self.z_dim))
        
        # 解码重建
        recon_x = self.decode(z_masked)
        
        return recon_x, q_m, q_v, z_dag
    
    def spatial_regularizer(self, z):
        """
        改进的空间正则化 - 使用距离衰减权重
        """
        # 获取HH和LL区域的索引
        hh_mask = (self.lisa_cluster == 1)  # HH聚类
        ll_mask = (self.lisa_cluster == 3)  # LL聚类
        
        # 计算距离矩阵
        dist_matrix = torch.cdist(self.coords, self.coords, p=2)

        
        # 应用高斯核函数（距离衰减）
        non_zero_dists = dist_matrix[dist_matrix > 0]
        bandwidth = torch.median(non_zero_dists) if len(non_zero_dists) > 0 else 1.0
        spatial_weights = torch.exp(-dist_matrix**2 / (2 * bandwidth**2 + 1e-8))

        # 计算HH区域的距离加权相似性
        hh_loss = 0
        if hh_mask.sum() > 1:
            hh_indices = torch.where(hh_mask)[0]
            hh_z = z[hh_indices]
            hh_weights = spatial_weights[hh_indices][:, hh_indices]
            weighted_mean = torch.sum(hh_weights.unsqueeze(-1) * hh_z.unsqueeze(1), dim=1) / hh_weights.sum(dim=1, keepdim=True)
            weighted_var = torch.sum(hh_weights * torch.sum((hh_z - weighted_mean)**2, dim=-1).unsqueeze(-1)) / hh_weights.sum()
            hh_loss = bandwidth * weighted_var # rescale by bandwith, could be adjusted
        # 计算LL区域的距离加权相似性
        ll_loss = 0
        if ll_mask.sum() > 1:
            ll_indices = torch.where(ll_mask)[0]
            ll_z = z[ll_indices]
            ll_weights = spatial_weights[ll_indices][:, ll_indices]
            # print(f"  ll_weights: {ll_weights}")
            weighted_mean = torch.sum(ll_weights.unsqueeze(-1) * ll_z.unsqueeze(1), dim=1) / ll_weights.sum(dim=1, keepdim=True)
            # print(f"  weighted_mean: {weighted_mean}")
            weighted_var = torch.sum(ll_weights * torch.sum((ll_z - weighted_mean)**2, dim=-1).unsqueeze(-1)) / ll_weights.sum()
            ll_loss = bandwidth * weighted_var # rescale by bandwith, could be adjusted
        # print(f"  ll_loss: {ll_loss}")
        # print(f"  hh_loss: {hh_loss}")
        
        
        return hh_loss + ll_loss
    
    def lag_regularizer(self, z):
        """
        空间滞后正则化 - 确保空间滞后特征被有效利用
        """
        # 计算空间滞后特征与潜变量的相关性
        spatial_lag_z = torch.sparse.mm(self.spatial_weight, z)
        correlation = F.cosine_similarity(z, spatial_lag_z, dim=1).mean()
        
        # 我们希望相关性较高，表示空间滞后特征被有效利用
        return 1.0 - correlation  # 最小化这个值使相关性接近1
    
    def negative_elbo_bound(self, x, spatial_lags, lambdav=0.001):
        """
        优化后的ELBO计算，包含空间正则化
        x: 城市特征数据 [batch_size, feature_dim]
        spatial_lags: 空间滞后特征 [batch_size, feature_dim]
        """
        # 前向传播
        recon_x, q_m, q_v, z_dag = self.forward(x, spatial_lags)
        
        # 重构损失 - 使用MSE
        rec = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
        
        # KL散度 - 潜变量分布约束
        p_m, p_v = torch.zeros_like(q_m), torch.ones_like(q_v)
        kl = kl_normal(q_m, q_v, p_m, p_v).mean()
        
        # DAG无环约束
        dag_reg = self.dag_regularizer()
        
        # 空间正则化 (HH/LL聚类)
        spatial_reg = self.spatial_regularizer(z_dag)
        
        # 空间滞后正则化
        lag_reg = self.lag_regularizer(z_dag)
        
        # 总损失
        nelbo = rec + kl + self.dag_weight * dag_reg + \
                self.spatial_reg_weight * spatial_reg + \
                self.lag_reg_weight * lag_reg
        # print(f"  Reconstruction: {rec.item():.4f}")
        # print(f"  KL Divergence: {kl.item():.4f}")
        # print(f"  DAG Regularization: {dag_reg.item():.4f} (weighted: {self.dag_weight * dag_reg.item():.4f})")
        # print(f"  Spatial Regularization: {spatial_reg.item():.9f} (weighted: {self.spatial_reg_weight * spatial_reg.item():.9f})")
        # print(f"  Lag Regularization: {lag_reg.item():.4f} (weighted: {self.lag_reg_weight * lag_reg.item():.4f})")
        # print(f"  Total NELBO: {nelbo.item():.4f}")
        
        
        
        return nelbo, kl, rec, recon_x, z_dag, spatial_reg, lag_reg
    
    def dag_regularizer(self):
        """DAG无环约束 - 确保因果结构有效"""
        return self.dag.notears_constraint()
    
    def loss(self, x, spatial_lags):
        nelbo, kl, rec, recon_x, z_dag, spatial_reg, lag_reg = self.negative_elbo_bound(x, spatial_lags)
        # 监控指标
        summaries = {
            'loss/total': nelbo.item(),
            'loss/kl': kl.item(),
            'loss/reconstruction': rec.item(),
            'loss/dag': self.dag_regularizer().item(),
            'loss/spatial_reg': spatial_reg.item(),
            'loss/lag_reg': lag_reg.item(),
            'metrics/mse': rec.item()  # 重建MSE
        }
        return nelbo, summaries
    
    def get_causal_structure(self, threshold=0.1):
        """获取学习到的因果结构矩阵（二值化）"""
        return self.dag.get_adjacency_matrix(threshold)
    
    def get_raw_causal_structure(self):
        """获取原始未二值化的因果结构矩阵（用于详细分析）"""
        return self.dag.get_raw_adjacency_matrix()
    
    def intervene(self, x, spatial_lags, intervention_type, **kwargs):
        """
        执行因果干预 - 支持因子干预和结构干预
        参数:
            intervention_type: 'factor' 或 'structural'
            kwargs: 根据干预类型提供不同参数
        """
        if intervention_type == 'factor':
            return self.factor_intervention(x, spatial_lags, **kwargs)
        elif intervention_type == 'structural':
            return self.structural_intervention(x, spatial_lags, **kwargs)
        else:
            raise ValueError(f"未知干预类型: {intervention_type}")
    
    def factor_intervention(self, x, spatial_lags, intervention_idx, intervention_values):
        """
        执行因子干预 (do操作) - 包含空间滞后特征
        支持向量干预（每个样本不同值）
        intervention_idx: 要干预的因果因子索引
        intervention_values: 干预值（标量或向量）
        """
        # 编码获取潜变量
        q_m, q_v = self.encode(x, spatial_lags)
        z = self.reparameterize(q_m, q_v)
        
        # 执行干预 (do操作)
        z_do = z.clone()
        
        # 支持标量和向量干预
        if isinstance(intervention_values, (int, float)):
            # 标量干预 - 所有样本设置相同值
            z_do[:, intervention_idx] = intervention_values
        else:
            # 向量干预 - 每个样本设置不同值
            if not isinstance(intervention_values, torch.Tensor):
                intervention_values = torch.tensor(intervention_values, dtype=z.dtype, device=z.device)
            z_do[:, intervention_idx] = intervention_values
        
        # 通过因果结构传播
        # 重塑为因果因子形式 [batch_size, z1_dim, z2_dim]
        z_reshaped = z_do.view(-1, self.z1_dim, self.z2_dim)
        # 应用DAG变换（注意：DAG层目前只支持z2_dim=1，所以这里squeeze）
        z_transformed = self.dag(z_reshaped.squeeze(-1))
        
        # 如果z2_dim>1，我们需要重新增加维度，但这里z2_dim=1，所以直接view
        z_transformed = z_transformed.view(-1, self.z_dim)
        
        # 解码反事实
        counterfactual = self.decode(z_transformed)
        
        return counterfactual, z_do
    
    def structural_intervention(self, x, spatial_lags, source_idx, target_idx, 
                                intervention_mode='remove', strength=0.0):
        """
        执行结构干预 - 修改因果结构
        """
        # 应用结构干预
        self.dag.apply_intervention(source_idx, target_idx, intervention_mode, strength)
        
        # 正常前向传播（使用新的结构）
        recon_x, q_m, q_v, z_dag = self.forward(x, spatial_lags)
        
        return recon_x, z_dag
    
    def simulate_policy_intervention(self, source_idx, target_idx, intervention_mode='remove', strength=0.0):
        """
        模拟结构干预政策
        参数:
            source_idx: 源变量索引
            target_idx: 目标变量索引
            intervention_mode: 干预类型 ('remove', 'strengthen', 'weaken', 'set')
            strength: 干预强度 (0.0-1.0)
        """
        # 应用干预到DAG层
        self.dag.apply_intervention(source_idx, target_idx, intervention_mode, strength)
    
    def structural_impact_analysis(self, interventions, X, spatial_lags, target_idx):
        """
        结构干预影响分析
        参数:
            interventions: 干预列表 [(source_idx, target_idx, intervention_mode, strength), ...]
            X: 输入特征
            spatial_lags: 空间滞后特征
            target_idx: 目标变量索引
        """
        device = next(self.parameters()).device
        X = X.to(device)
        spatial_lags = spatial_lags.to(device)
        
        # 记录原始状态
        original_structure = self.dag.A.data.clone()
        
        # 获取原始输出
        with torch.no_grad():
            original_output = self.forward(X, spatial_lags)[0][:, target_idx]
        
        # 应用所有干预
        for src, tgt, itype, strength in interventions:
            self.simulate_policy_intervention(src, tgt, itype, strength)
        
        # 获取干预后输出
        with torch.no_grad():
            intervened_output = self.forward(X, spatial_lags)[0][:, target_idx]
        
        # 恢复原始结构
        self.dag.A.data = original_structure
        
        # 计算变化
        change = intervened_output - original_output
        return change.detach().cpu()
    
    def calculate_residuals(self, x, spatial_lags):
        """计算残差用于空间自相关分析"""
        with torch.no_grad():
            recon_x, _, _, _, _ = self.negative_elbo_bound(x, spatial_lags)
            return x - recon_x