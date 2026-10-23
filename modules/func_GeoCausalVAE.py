import run_options as options
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import os
os.environ['USE_PYGEOS'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ====================== 核心因果推断组件 ======================
class DagLayer(nn.Module):
    """有向无环图层，A[source, target]"""
    def __init__(self, in_features, out_features, bias=False, initial=True, prior_dag=None):
        super().__init__()
        if in_features != out_features or bias:
            raise ValueError("Eq. (3) requires a square adjacency and no separate bias.")
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('I', torch.eye(out_features))
        # 不把经验先验强制当作DAG。
        self.A = nn.Parameter(0.01 * torch.randn(out_features, out_features) * (1 - self.I))
        if prior_dag is not None:
            prior_dag = torch.as_tensor(prior_dag, dtype=torch.float32).clone()
            if prior_dag.shape != self.I.shape or not torch.isfinite(prior_dag).all():
                raise ValueError("prior_dag must be a finite square matrix in model variable order.")
            prior_dag = prior_dag * (1 - self.I)
        self.register_buffer('prior_dag', prior_dag)
        self.register_parameter('bias', None)

    def processed_A(self):
        return self.A * (1 - self.I)

    def forward(self, epsilon):
        # (I - A^T) z = epsilon；。
        return torch.linalg.solve(self.I - self.processed_A().T, epsilon.T).T

    def get_adjacency_matrix(self, threshold=0.1):
        return (self.processed_A().abs() > threshold).float().detach().cpu().numpy()

    def get_raw_adjacency_matrix(self):
        return self.processed_A().detach().cpu().numpy().copy()

    def notears_constraint(self):
        A = self.processed_A().double()
        return (torch.trace(torch.matrix_exp(A * A)) - self.out_features).to(self.A.dtype)

    def prior_constraint(self, u):
        # 。
        if self.prior_dag is None:
            return u.new_zeros(())
        if u.shape[1] != self.out_features:
            raise ValueError("Graph supervision signals must match adjacency dimension.")
        difference = torch.sigmoid(u @ self.processed_A()) - u @ self.prior_dag
        return difference.square().sum(dim=1).mean()

    def graph_diagnostics(self):
        """检查图的无环性和数值稳定性"""
        A = self.get_raw_adjacency_matrix()
        finite = bool(np.isfinite(A).all())
        if not finite:
            return {'finite': False, 'is_dag': False, 'notears_h': None,
                    'edge_threshold': 0.0, 'postprocessing': 'none'}
        edges = A != 0
        indegree = edges.sum(axis=0)
        available = list(np.flatnonzero(indegree == 0))
        visited = 0
        while available:
            source = available.pop()
            visited += 1
            for target in np.flatnonzero(edges[source]):
                indegree[target] -= 1
                if indegree[target] == 0:
                    available.append(int(target))
        with torch.no_grad():
            h = float(self.notears_constraint().cpu())
        return {'finite': True, 'is_dag': visited == self.out_features,
                'notears_h': h if np.isfinite(h) else None,
                'edge_threshold': 0.0, 'postprocessing': 'none'}

    def project_to_dag(self):

        A = self.get_raw_adjacency_matrix()
        if not np.isfinite(A).all():
            raise ValueError("Non-finite adjacency cannot be projected.")
        reach = np.eye(self.out_features, dtype=bool)
        keep = np.zeros_like(A)
        edges = [(i, j) for i in range(self.out_features)
                 for j in range(self.out_features) if i != j and A[i, j] != 0]
        edges.sort(key=lambda edge: (-abs(A[edge]), edge[0], edge[1]))
        removed = []
        for i, j in edges:
            if reach[j, i]:
                removed.append([i, j, float(A[i, j])])
                continue
            keep[i, j] = A[i, j]
            reach |= reach[:, i:i+1] & reach[j:j+1, :]
        with torch.no_grad():
            self.A.copy_(torch.as_tensor(keep, dtype=self.A.dtype, device=self.A.device))
        return removed

    def apply_intervention(self, source_idx, target_idx, intervention_type='remove', strength=0.0):
        assert 0 <= source_idx < self.out_features and 0 <= target_idx < self.out_features
        if source_idx == target_idx or not np.isfinite(strength):
            raise ValueError("A structural intervention requires a finite, off-diagonal edge.")
        current_weight = self.A[source_idx, target_idx].item()
        if intervention_type == 'remove':
            new_weight = 0.0
        elif intervention_type == 'set':
            new_weight = float(strength)
        elif intervention_type in ('strengthen', 'weaken'):
            if options.LIMIT_INTERVENTION_STRENGTH and not 0 <= strength <= 1:
                raise ValueError("Relative intervention strength must be in [0, 1].")
            new_weight = current_weight * (1 + strength if intervention_type == 'strengthen' else 1 - strength)
        else:
            raise ValueError(f"未知干预类型: {intervention_type}")

        with torch.no_grad():
            self.A[source_idx, target_idx] = new_weight
        print(f"干预: {source_idx}→{target_idx} | 权重: {current_weight:.4f} → {new_weight:.4f}")
class SimplifiedMask(nn.Module):
    """Eq. (4) 的逐节点父变量掩码，输入包含空间滞后 Lz"""
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * z_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            ) for _ in range(z_dim)
        ])

    def forward(self, z, A, spatial_z, epsilon):
        # z 为线性SCM输出；z_masked 为非线性重构。
        outputs = []
        for i in range(self.z_dim):
            parents = z * A[:, i].unsqueeze(0)
            inputs = torch.cat([parents, spatial_z], dim=1)
            outputs.append(self.net[i](inputs) + epsilon[:, i:i+1])
        return torch.cat(outputs, dim=1)

class ConditionalPrior(nn.Module):

    def __init__(self, u_dim, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(u_dim, 32),
            nn.ELU(),
            nn.Linear(32, 2 * z_dim)
        )

    def forward(self, u):
        m, v = self.net(u).chunk(2, dim=-1)
        return m, F.softplus(v) + 1e-8
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
    """带DAG结构的表格数据解码器 - 重建原始特征"""
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
    def __init__(self, input_dim, spatial_weight_matrix, lisa_cluster, coords, z1_dim=13, z2_dim=1, 
                 spatial_reg_weight=0.01, lag_reg_weight=0.0, dag_weight=0.5, prior_dag=None, prior_dag_weight=1.0, use_dag=True):
        """
        参数:
            input_dim: 输入特征维度
            spatial_weight_matrix: 空间权重矩阵 (PyTorch张量)
            lisa_cluster: LISA聚类结果 (numpy数组)
            z1_dim: 因果因子数量
            z2_dim: 每个因果因子的维度 (设为1，标量)
            spatial_reg_weight: 空间正则化权重
            lag_reg_weight: 空间滞后正则化权重
            prior_dag: 先验DAG矩阵 (二值矩阵)
            prior_dag_weight: 先验DAG正则化权重
        """
        super().__init__()
        self.input_dim = input_dim
        self.use_dag = use_dag
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.z_dim = z1_dim * z2_dim  # 总潜变量维度
        if z2_dim != 1 or self.z_dim != input_dim:
            raise ValueError("This manuscript uses one scalar latent per reconstructed variable.")
        if lag_reg_weight != 0 and not options.ALLOW_EXTRA_LAG_PENALTY:
            raise ValueError("no separate lag penalty.")
        if use_dag and prior_dag is None and prior_dag_weight != 0:
            raise ValueError("The main model requires the empirical prior.")
        self.u_dim = input_dim - 1
        self.conditional_prior = ConditionalPrior(self.u_dim, self.z_dim)
        self.spatial_reg_weight = spatial_reg_weight
        self.lag_reg_weight = lag_reg_weight
        self.dag_weight = dag_weight
        self.prior_dag_weight = prior_dag_weight
        self.register_buffer('coords', coords)
        
        # 存储空间关系数据
        self.register_buffer('spatial_weight', spatial_weight_matrix)
        self.register_buffer('lisa_cluster', torch.tensor(lisa_cluster, dtype=torch.long))
        
        # 编码器 - 包含空间滞后特征
        self.enc = SpatialAwareTabularEncoder(input_dim, self.z_dim)
        
        # 解码器
        self.dec = TabularDecoderDAG(self.z_dim, input_dim)
        
        # DAG层 - 传入先验DAG
        self.dag = DagLayer(self.z1_dim, self.z1_dim, prior_dag=prior_dag)
        
        # 潜变量掩码层
        self.mask_z = SimplifiedMask(self.z_dim)
        if not self.use_dag:

            with torch.no_grad():
                self.dag.A.zero_()
            self.dag.requires_grad_(False)
            self.mask_z.requires_grad_(False)
        
        # 空间聚类嵌入层 (用于HH/LL聚类正则化)
        # HH/LL labels are used directly by spatial_regularizer.
    
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
        if not self.use_dag:
            return z
        # 重塑为因果因子形式 [batch_size, z1_dim, z2_dim]
        z_reshaped = z.view(-1, self.z1_dim, self.z2_dim)
        
        # 应用DAG变换
        z_dag = self.dag(z_reshaped.squeeze(-1))  # squeeze(-1) 因为 z2_dim=1
        
        return z_dag
    
    def decode(self, z):
        """从潜变量重建数据"""
        return self.dec(z)
    
    def forward(self, x, spatial_lags, sample=True):

        q_m, q_v = self.encode(x, spatial_lags)
        epsilon = self.reparameterize(q_m, q_v) if sample else q_m
        z_dag = self.apply_dag(epsilon)
        spatial_z = torch.sparse.mm(self.spatial_weight, z_dag)
        z_masked = self.mask_z(z_dag, self.dag.processed_A(), spatial_z, epsilon) if self.use_dag else z_dag
        recon_x = self.decode(z_masked)
        return recon_x, q_m, q_v, z_dag
    
    def spatial_regularizer(self, z):

        dist_matrix = torch.cdist(self.coords, self.coords, p=2)
        non_zero_dists = dist_matrix[dist_matrix > 0]
        bandwidth = torch.median(non_zero_dists) if non_zero_dists.numel() else z.new_tensor(1.0)
        spatial_weights = torch.exp(-dist_matrix.square() / (2 * bandwidth.square()))
        hh = self.lisa_cluster == 1
        ll = self.lisa_cluster == 3
        same_cluster = (hh[:, None] & hh[None, :]) | (ll[:, None] & ll[None, :])
        latent_distance = (z[:, None, :] - z[None, :, :]).square().sum(dim=-1)
        return (spatial_weights * same_cluster * latent_distance).sum()
    
    def lag_regularizer(self, z):
        """
        空间滞后正则化 
        """
        # 计算空间滞后特征与潜变量的相关性
        spatial_lag_z = torch.sparse.mm(self.spatial_weight, z)
        correlation = F.cosine_similarity(z, spatial_lag_z, dim=1).mean()
        
        # 我们希望相关性较高，表示空间滞后特征被有效利用
        return 1.0 - correlation  # 最小化这个值使相关性接近1
    
    def negative_elbo_bound(self, x, spatial_lags, lambdav=0.001):

        recon_x, q_m, q_v, z_dag = self.forward(x, spatial_lags)
        rec = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
        kl_eps = kl_normal(q_m, q_v, torch.zeros_like(q_m), torch.ones_like(q_v)).mean()


        u = x[:, :self.u_dim]
        p_m, p_v = self.conditional_prior(u)
        C = torch.linalg.solve(self.dag.I - self.dag.processed_A().T, self.dag.I)
        z_mean = q_m @ C.T
        z_variance = q_v @ C.square().T
  
        _, logabsdet_C = torch.linalg.slogdet(C)
        logdet_q = torch.log(q_v).sum(dim=1) + 2 * logabsdet_C
        kl_z = 0.5 * (
            (z_variance / p_v + (z_mean - p_m).square() / p_v).sum(dim=1)
            - self.z_dim + torch.log(p_v).sum(dim=1) - logdet_q
        ).mean()
        kl = kl_eps + kl_z

        dag_reg = self.dag_regularizer()

        u_graph = torch.cat([u, u.new_zeros((u.size(0), 1))], dim=1)
        prior_reg = self.dag_prior_constraint(u_graph)
        spatial_reg = self.spatial_regularizer(z_dag)
        lag_reg = self.lag_regularizer(z_dag) if self.lag_reg_weight != 0 else x.new_zeros(())
        nelbo = rec + kl + self.dag_weight * dag_reg + \
                self.prior_dag_weight * prior_reg + self.spatial_reg_weight * spatial_reg + self.lag_reg_weight * lag_reg
        if not torch.isfinite(nelbo):
            raise FloatingPointError("Non-finite training objective.")
        return nelbo, kl, rec, recon_x, z_dag, spatial_reg, lag_reg, prior_reg
    
    def dag_regularizer(self):
        """DAG无环约束 """
        return self.dag.notears_constraint()
    
    def dag_prior_constraint(self, u):
        return self.dag.prior_constraint(u) if self.use_dag else u.new_zeros(())
    
    def loss(self, x, spatial_lags):
        nelbo, kl, rec, recon_x, z_dag, spatial_reg, lag_reg, prior_reg = self.negative_elbo_bound(x, spatial_lags)
        # 监控指标
        summaries = {
            'loss/total': nelbo.item(),
            'loss/kl': kl.item(),
            'loss/reconstruction': rec.item(),
            'loss/dag': self.dag_regularizer().item(),
            'loss/prior_dag': prior_reg.item(),
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
    
    def factor_intervention(self, x, spatial_lags, intervention_idx, intervention_values, intervention_type='set'):

        if not 0 <= intervention_idx < self.z_dim:
            raise IndexError("Invalid latent intervention index.")
        q_m, q_v = self.encode(x, spatial_lags)
        epsilon = self.reparameterize(q_m, q_v)
        z_base = self.apply_dag(epsilon)
        value = torch.as_tensor(intervention_values, device=x.device, dtype=x.dtype)
        value = torch.broadcast_to(value, (x.size(0),))
        if intervention_type == 'add':
            value = z_base[:, intervention_idx] + value
        elif intervention_type == 'scale':
            value = z_base[:, intervention_idx] * value
        elif intervention_type != 'set':
            raise ValueError(f"Unsupported intervention type: {intervention_type}")
        A_do = self.dag.processed_A().clone()
        A_do[:, intervention_idx] = 0  # 删除进入被干预节点的边。
        epsilon_do = epsilon.clone()
        epsilon_do[:, intervention_idx] = value
        z_do = torch.linalg.solve(self.dag.I - A_do.T, epsilon_do.T).T
        spatial_z = torch.sparse.mm(self.spatial_weight, z_do)
        z_masked = self.mask_z(z_do, A_do, spatial_z, epsilon_do) if self.use_dag else z_do.clone()
        z_masked[:, intervention_idx] = value
        return self.decode(z_masked), z_do
    
    
    def structural_intervention(self, x, spatial_lags, source_idx, target_idx,
                                intervention_mode='remove', strength=0.0):
        original = self.dag.A.detach().clone()
        try:
            self.dag.apply_intervention(source_idx, target_idx, intervention_mode, strength)
            recon_x, _, _, z_dag = self.forward(x, spatial_lags, sample=False)
            return recon_x, z_dag
        finally:
            with torch.no_grad():
                self.dag.A.copy_(original)
    
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
        device = next(self.parameters()).device
        X, spatial_lags = X.to(device), spatial_lags.to(device)
        original = self.dag.A.detach().clone()
        try:
            with torch.no_grad():
                original_output = self.forward(X, spatial_lags, sample=False)[0][:, target_idx]
                for src, tgt, itype, strength in interventions:
                    self.simulate_policy_intervention(src, tgt, itype, strength)
                changed_output = self.forward(X, spatial_lags, sample=False)[0][:, target_idx]
                return (changed_output - original_output).cpu()
        finally:
            with torch.no_grad():
                self.dag.A.copy_(original)
    
    def calculate_residuals(self, x, spatial_lags):
        with torch.no_grad():
            recon_x = self.forward(x, spatial_lags, sample=False)[0]
            return x - recon_x
