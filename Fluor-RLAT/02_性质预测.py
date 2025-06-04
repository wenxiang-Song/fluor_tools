#######################################################################
# 预测吸收波长
import pandas as pd
import numpy as np
import os
import random
import copy
import dgl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from dgllife.model import model_zoo
from dgllife.utils import smiles_to_bigraph
from dgllife.utils import EarlyStopping, Meter
from dgllife.utils import AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer
from dgllife.data import MoleculeCSVDataset
from dgllife.model.gnn import AttentiveFPGNN
from dgllife.model.readout import AttentiveFPReadout
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.neighbors import KernelDensity


if torch.cuda.is_available():
    print('use GPU')
    device = 'cuda'
else:
    print('use CPU')
    device = 'cpu'

# 设置全局随机种子
seed = 42
alpha = 0.1
epochs = 3
patience = 20
n_tasks = 1
graph_feat_size = 256 # 固定
batch_size = 32       # 固定
learning_rate = 1e-3  # 固定


dropout = 0.3
num_layers = 2
num_timesteps = 2 


random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# 使用 AttentiveFP featurizer
atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='hv')
bond_featurizer = AttentiveFPBondFeaturizer(bond_data_field='he')
n_feats = atom_featurizer.feat_size('hv')
e_feats = bond_featurizer.feat_size('he')
print("n_feats", n_feats, "e_feats", e_feats)

node_feat_size = n_feats
edge_feat_size = e_feats

def compute_lds_weights(targets, h=alpha, sigma=5, sqrt=False, amplify=False):
    targets = np.array(targets).reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=sigma).fit(targets)
    log_densities = kde.score_samples(targets)
    densities = np.exp(log_densities)
    weights = 1. / (densities ** h)
    if sqrt:
        weights = np.sqrt(weights)
    if amplify:
        median_val = np.median(targets)
        weights *= np.where(np.abs(targets - median_val) > 1.5, 2.0, 1.0)
    return torch.tensor(weights / np.mean(weights), dtype=torch.float32)


def load_data_with_fp(data, fp_data, name, load):
    dataset = MoleculeCSVDataset(data,
                                 smiles_to_graph=smiles_to_bigraph,
                                 node_featurizer=atom_featurizer,
                                 edge_featurizer=bond_featurizer,
                                 smiles_column='smiles',
                                 cache_file_path=str(name)+'_dataset_abs.bin',
                                 task_names=['abs'],
                                 load=load, init_mask=True, n_jobs=1)

    combined_data = []
    for i, data_tuple in enumerate(dataset):
        if len(data_tuple) == 3:
            smiles, graph, label = data_tuple
            mask = None
        else:
            smiles, graph, label, mask = data_tuple
        fp = torch.tensor(fp_data[i], dtype=torch.float32)
        combined_data.append((graph, fp, label, mask))
    return combined_data


#指纹数据加载
def load_fingerprints(fp_file):
    df = pd.read_csv(fp_file)
    return torch.tensor(df.values, dtype=torch.float32)



#数据加载
train_data = pd.read_csv('./data/train_abs.csv')
valid_data = pd.read_csv('./data/valid_abs.csv')

# #数据标准化
scaler = StandardScaler()
train_data[['abs']] = scaler.fit_transform(train_data[['abs']])
valid_data[['abs']] = scaler.transform(valid_data[['abs']])

train_fp_solvent = load_fingerprints('./data/train_sol_abs.csv')
valid_fp_solvent = load_fingerprints('./data/valid_sol_abs.csv')
train_fp_smiles = load_fingerprints('./data/train_smiles_abs.csv')
valid_fp_smiles = load_fingerprints('./data/valid_smiles_abs.csv')

# === 从 train_data / valid_data 中提取额外特征（列索引 8:152）===
train_fp_extra = torch.tensor(train_data.iloc[:, 8:152].values, dtype=torch.float32)
valid_fp_extra = torch.tensor(valid_data.iloc[:, 8:152].values, dtype=torch.float32)

# === 数值部分（8列）归一化 ===
scaler_num = MinMaxScaler()

# 拆分：前 8 列为数值特征，后面为补充指纹
train_num = train_fp_extra[:, :8].numpy()
valid_num = valid_fp_extra[:, :8].numpy()

train_rest = train_fp_extra[:, 8:]  # tensor 后部分
valid_rest = valid_fp_extra[:, 8:]

# 拟合并归一化前8列
train_num_scaled = scaler_num.fit_transform(train_num)
valid_num_scaled = scaler_num.transform(valid_num)

# 转换回 tensor 并拼接
train_fp_extra = torch.cat([torch.tensor(train_num_scaled, dtype=torch.float32), train_rest], dim=1)
valid_fp_extra = torch.cat([torch.tensor(valid_num_scaled, dtype=torch.float32), valid_rest], dim=1)

# === 拼接最终特征：solvent + smiles + extra ===
train_fp = torch.cat([train_fp_solvent, train_fp_smiles, train_fp_extra], dim=1)
valid_fp = torch.cat([valid_fp_solvent, valid_fp_smiles, valid_fp_extra], dim=1)


lds_weights = compute_lds_weights(train_data[['abs']].values.flatten())


class FingerprintAttentionCNN(nn.Module):
    def __init__(self, input_dim, conv_channels=64):
        super(FingerprintAttentionCNN, self).__init__()
        self.conv_feat = nn.Conv1d(1, conv_channels, kernel_size=3, padding=1)
        self.conv_attn = nn.Conv1d(1, conv_channels, kernel_size=3, padding=1)
        self.softmax = nn.Softmax(dim=-1)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, D]
        feat_map = self.conv_feat(x)         # [B, C, D]
        attn_map = self.conv_attn(x)         # [B, C, D]
        attn_weights = self.softmax(attn_map)
        attn_out = torch.sum(feat_map * attn_weights, dim=-1)  # [B, C]
        pooled = self.pool(feat_map).squeeze(-1)               # [B, C]
        return torch.cat([attn_out, pooled], dim=1)            # [B, 2C]



class GraphFingerprintsModel(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size,
                solvent_dim, smiles_extra_dim,  # 分开声明两个 fp 输入维度
                graph_feat_size=graph_feat_size, num_layers=num_layers, num_timesteps=num_timesteps,
                n_tasks=n_tasks, dropout=dropout):
        super(GraphFingerprintsModel, self).__init__()

        # 图神经网络部分
        self.gnn = AttentiveFPGNN(node_feat_size=node_feat_size,
                                edge_feat_size=edge_feat_size,
                                num_layers=num_layers,
                                graph_feat_size=graph_feat_size,
                                dropout=dropout)
        self.readout = AttentiveFPReadout(feat_size=graph_feat_size,
                                        num_timesteps=num_timesteps,
                                        dropout=dropout)

        # 指纹部分一：smiles + extra，使用 CNN-attention 提取
        self.fp_extractor = FingerprintAttentionCNN(smiles_extra_dim, conv_channels=graph_feat_size)

        # 指纹部分二：solvent，使用全连接提取
        self.solvent_extractor = nn.Sequential(
            nn.Linear(solvent_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, graph_feat_size)
        )


        total_input_dim = graph_feat_size + graph_feat_size + 2 * graph_feat_size
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_tasks)
        )




    def forward(self, g, node_feats, edge_feats, fingerprints):
        if edge_feats is None or 'he' not in g.edata:
            num_edges = g.number_of_edges()
            edge_feats = torch.zeros((num_edges, edge_feats.size(1)), device=g.device)

        node_feats = self.gnn(g, node_feats, edge_feats)
        graph_feats = self.readout(g, node_feats)  # [B, G]

        # === 分离 fingerprints 三部分 ===
        # 假设 fingerprints.shape = [B, S + M + E]（solvent, smiles, extra）
        # 你可以根据各自维度切分
        B = fingerprints.size(0)
        solvent_feat = fingerprints[:, :train_fp_solvent.shape[1]]  # [B, S]
        smiles_extra_feat = fingerprints[:, train_fp_solvent.shape[1]:]  # [B, M+E]

        # 分别提取特征
        solvent_out = self.solvent_extractor(solvent_feat)  # [B, G]
        smiles_extra_out = self.fp_extractor(smiles_extra_feat)  # [B, 2G]

        # 拼接三部分特征
        combined_feats = torch.cat([graph_feats, solvent_out, smiles_extra_out], dim=1)  # [B, 3G]

        return self.predict(combined_feats)

# 自定义数据集类
class MolecularDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 数据加载的collate函数
def collate_fn(batch):
    if len(batch[0]) == 5:
        graphs, fps, labels, masks, weights = zip(*batch)
        weights = torch.stack(weights)
    else:
        graphs, fps, labels, masks = zip(*batch)
        weights = None
    graphs = dgl.batch(graphs)
    fps = torch.stack(fps)
    labels = torch.stack(labels)
    masks = torch.stack(masks) if masks[0] is not None else None
    return graphs, fps, labels, masks, weights



# === 读取 target 数据 ===
target_data = pd.read_csv('./input/input.csv')
target_fp_solvent = load_fingerprints('./input/target_sol_morgan.csv')
target_fp_smiles = load_fingerprints('./input/target_smiles_morgan.csv')

# === 数值特征标准化 ===
target_fp_extra = torch.tensor(target_data.iloc[:, 8:152].values, dtype=torch.float32)
target_num = target_fp_extra[:, :8].numpy()
target_rest = target_fp_extra[:, 8:]

# 加载预训练的 scaler_num
target_num_scaled = scaler_num.transform(target_num)
target_fp_extra = torch.cat([torch.tensor(target_num_scaled, dtype=torch.float32), target_rest], dim=1)

# === 拼接最终指纹 ===
target_fp = torch.cat([target_fp_solvent, target_fp_smiles, target_fp_extra], dim=1)

# === 标签标准化（只为保持接口一致，其实预测时不需操作标签） ===
target_data[['abs']] = scaler.transform(target_data[['abs']])  # 注意：仅为构造 dataset，不影响预测结果

# === 构造 dataset 与 dataloader ===
target_datasets = load_data_with_fp(target_data, target_fp, 'target', True)
target_dataset = MolecularDataset(target_datasets)
target_loader = DataLoader(target_dataset, batch_size=batch_size, collate_fn=collate_fn)




# 初始化模型
solvent_dim = target_fp_solvent.shape[1]
smiles_extra_dim = target_fp_smiles.shape[1] + target_fp_extra.shape[1]

model = GraphFingerprintsModel(
    node_feat_size=n_feats,
    edge_feat_size=e_feats,
    solvent_dim=solvent_dim,
    smiles_extra_dim=smiles_extra_dim,
    graph_feat_size=graph_feat_size,
    num_layers=num_layers,
    num_timesteps=num_timesteps,
    n_tasks=n_tasks,
    dropout=dropout
).to(device)

# 加载保存的模型参数
model.load_state_dict(torch.load('Model_abs.pth', map_location=device))
model.eval()


# 预测
def predict(model, dataloader):
    all_predictions = []
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 5:
                graphs, fps, _, _, _ = batch  # 包含权重
            else:
                graphs, fps, _, _ = batch     # 不含权重
            graphs = graphs.to(device)
            fps = fps.to(device)
            node_feats = graphs.ndata['hv']
            edge_feats = graphs.edata['he']
            predictions = model(graphs, node_feats, edge_feats, fps)
            all_predictions.append(predictions.cpu().numpy())
    return np.vstack(all_predictions)


# 将预测结果保存到 CSV 文件
def save_predictions(predictions, file_name):
    df = pd.DataFrame(predictions, columns=['abs'])
    df.to_csv(file_name, index=False)

# 预测完成后，反向转换标准化的预测结果
def reverse_standardization(predictions, scaler):
    return scaler.inverse_transform(predictions)


# === 模型预测 ===
target_predictions = predict(model, target_loader)
target_scale_predictions = reverse_standardization(target_predictions, scaler)

# === 保存预测结果 ===
save_predictions(target_scale_predictions, './result/target_predictions_abs.csv')
print("🎯 Target predictions saved to target_predictions_abs.csv")






#######################################################################
# 预测发射波长

graph_feat_size = 256
alpha = 0
num_layers = 3
num_timesteps = 1
n_tasks = 1
dropout = 0.3
batch_size = 32
learning_rate = 1e-3
epochs = 3
patience = 20


random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)



def load_data_with_fp(data, fp_data, name, load):
    dataset = MoleculeCSVDataset(data,
                                 smiles_to_graph=smiles_to_bigraph,
                                 node_featurizer=atom_featurizer,
                                 edge_featurizer=bond_featurizer,
                                 smiles_column='smiles',
                                 cache_file_path=str(name)+'_dataset_em.bin',
                                 task_names=['em'],
                                 load=load, init_mask=True, n_jobs=1)

    combined_data = []
    for i, data_tuple in enumerate(dataset):
        if len(data_tuple) == 3:
            smiles, graph, label = data_tuple
            mask = None
        else:
            smiles, graph, label, mask = data_tuple
        fp = torch.tensor(fp_data[i], dtype=torch.float32)
        combined_data.append((graph, fp, label, mask))
    return combined_data




#数据加载
train_data = pd.read_csv('./data/train_em.csv')
valid_data = pd.read_csv('./data/valid_em.csv')

#数据标准化
scaler = StandardScaler()
train_data[['em']] = scaler.fit_transform(train_data[['em']])
valid_data[['em']] = scaler.transform(valid_data[['em']])

train_fp_solvent = load_fingerprints('./data/train_sol_em.csv')
valid_fp_solvent = load_fingerprints('./data/valid_sol_em.csv')
train_fp_smiles = load_fingerprints('./data/train_smiles_em.csv')
valid_fp_smiles = load_fingerprints('./data/valid_smiles_em.csv')

# === 从 train_data / valid_data 中提取额外特征（列索引 8:152）===
train_fp_extra = torch.tensor(train_data.iloc[:, 8:152].values, dtype=torch.float32)
valid_fp_extra = torch.tensor(valid_data.iloc[:, 8:152].values, dtype=torch.float32)

# === 数值部分（8列）归一化 ===
scaler_num = MinMaxScaler()

# 拆分：前 8 列为数值特征，后面为补充指纹
train_num = train_fp_extra[:, :8].numpy()
valid_num = valid_fp_extra[:, :8].numpy()

train_rest = train_fp_extra[:, 8:]  # tensor 后部分
valid_rest = valid_fp_extra[:, 8:]

# 拟合并归一化前8列
train_num_scaled = scaler_num.fit_transform(train_num)
valid_num_scaled = scaler_num.transform(valid_num)

# 转换回 tensor 并拼接
train_fp_extra = torch.cat([torch.tensor(train_num_scaled, dtype=torch.float32), train_rest], dim=1)
valid_fp_extra = torch.cat([torch.tensor(valid_num_scaled, dtype=torch.float32), valid_rest], dim=1)

# === 拼接最终特征：solvent + smiles + extra ===
train_fp = torch.cat([train_fp_solvent, train_fp_smiles, train_fp_extra], dim=1)
valid_fp = torch.cat([valid_fp_solvent, valid_fp_smiles, valid_fp_extra], dim=1)
lds_weights = compute_lds_weights(train_data[['em']].values.flatten())



class GraphFingerprintsModel(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size,
                 solvent_dim, smiles_extra_dim,  # 分开声明两个 fp 输入维度
                 graph_feat_size=graph_feat_size, num_layers=num_layers, num_timesteps=num_timesteps,
                 n_tasks=n_tasks, dropout=dropout):
        super(GraphFingerprintsModel, self).__init__()

        # 图神经网络部分
        self.gnn = AttentiveFPGNN(node_feat_size=node_feat_size,
                                  edge_feat_size=edge_feat_size,
                                  num_layers=num_layers,
                                  graph_feat_size=graph_feat_size,
                                  dropout=dropout)
        self.readout = AttentiveFPReadout(feat_size=graph_feat_size,
                                          num_timesteps=num_timesteps,
                                          dropout=dropout)

        # 指纹部分一：smiles + extra，使用 CNN-attention 提取
        self.fp_extractor = FingerprintAttentionCNN(smiles_extra_dim, conv_channels=graph_feat_size)

        # 指纹部分二：solvent，使用全连接提取
        self.solvent_extractor = nn.Sequential(
            nn.Linear(solvent_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, graph_feat_size)
        )

        total_input_dim = graph_feat_size + graph_feat_size + 2 * graph_feat_size
        # 最终拼接后预测（3 * graph_feat_size）
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_tasks)
        )

    def forward(self, g, node_feats, edge_feats, fingerprints):
        if edge_feats is None or 'he' not in g.edata:
            num_edges = g.number_of_edges()
            edge_feats = torch.zeros((num_edges, edge_feats.size(1)), device=g.device)

        node_feats = self.gnn(g, node_feats, edge_feats)
        graph_feats = self.readout(g, node_feats)  # [B, G]

        # === 分离 fingerprints 三部分 ===
        # 假设 fingerprints.shape = [B, S + M + E]（solvent, smiles, extra）
        # 你可以根据各自维度切分
        B = fingerprints.size(0)
        solvent_feat = fingerprints[:, :train_fp_solvent.shape[1]]  # [B, S]
        smiles_extra_feat = fingerprints[:, train_fp_solvent.shape[1]:]  # [B, M+E]

        # 分别提取特征
        solvent_out = self.solvent_extractor(solvent_feat)  # [B, G]
        smiles_extra_out = self.fp_extractor(smiles_extra_feat)  # [B, 2G]

        # 拼接三部分特征
        combined_feats = torch.cat([graph_feats, solvent_out, smiles_extra_out], dim=1)  # [B, 3G]

        return self.predict(combined_feats)



# === 读取 target 数据 ===
target_data = pd.read_csv('./input/input.csv')
target_fp_solvent = load_fingerprints('./input/target_sol_morgan.csv')
target_fp_smiles = load_fingerprints('./input/target_smiles_morgan.csv')

# === 数值特征标准化 ===
target_fp_extra = torch.tensor(target_data.iloc[:, 8:152].values, dtype=torch.float32)
target_num = target_fp_extra[:, :8].numpy()
target_rest = target_fp_extra[:, 8:]

# 加载预训练的 scaler_num
target_num_scaled = scaler_num.transform(target_num)
target_fp_extra = torch.cat([torch.tensor(target_num_scaled, dtype=torch.float32), target_rest], dim=1)

# === 拼接最终指纹 ===
target_fp = torch.cat([target_fp_solvent, target_fp_smiles, target_fp_extra], dim=1)

# === 标签标准化（只为保持接口一致，其实预测时不需操作标签） ===
target_data[['em']] = scaler.transform(target_data[['em']])  # 注意：仅为构造 dataset，不影响预测结果

# === 构造 dataset 与 dataloader ===
target_datasets = load_data_with_fp(target_data, target_fp, 'target', True)
target_dataset = MolecularDataset(target_datasets)
target_loader = DataLoader(target_dataset, batch_size=batch_size, collate_fn=collate_fn)




# 初始化模型
solvent_dim = target_fp_solvent.shape[1]
smiles_extra_dim = target_fp_smiles.shape[1] + target_fp_extra.shape[1]

model = GraphFingerprintsModel(
    node_feat_size=n_feats,
    edge_feat_size=e_feats,
    solvent_dim=solvent_dim,
    smiles_extra_dim=smiles_extra_dim,
    graph_feat_size=graph_feat_size,
    num_layers=num_layers,
    num_timesteps=num_timesteps,
    n_tasks=n_tasks,
    dropout=dropout
).to(device)

# 加载保存的模型参数
model.load_state_dict(torch.load('Model_em.pth', map_location=device))
model.eval()



# 将预测结果保存到 CSV 文件
def save_predictions(predictions, file_name):
    df = pd.DataFrame(predictions, columns=['em'])
    df.to_csv(file_name, index=False)


# === 模型预测 ===
target_predictions = predict(model, target_loader)
target_scale_predictions = reverse_standardization(target_predictions, scaler)

# === 保存预测结果 ===
save_predictions(target_scale_predictions, './result/target_predictions_em.csv')
print("🎯 Target predictions saved to target_predictions_em.csv")






#######################################################################
# 预测量子产率

graph_feat_size = 256
n_tasks = 1
dropout = 0.4
alpha = 0.2
batch_size = 32
learning_rate = 1e-3
epochs = 3
patience = 20
num_layers = 2
num_timesteps = 3

random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)




atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='hv')
bond_featurizer = AttentiveFPBondFeaturizer(bond_data_field='he')
n_feats = atom_featurizer.feat_size('hv')
e_feats = bond_featurizer.feat_size('he')


def load_data_with_fp(data, fp_data, name, load):
    dataset = MoleculeCSVDataset(data,
                                 smiles_to_graph=smiles_to_bigraph,
                                 node_featurizer=atom_featurizer,
                                 edge_featurizer=bond_featurizer,
                                 smiles_column='smiles',
                                 cache_file_path=str(name)+'_dataset_plqy.bin',
                                 task_names=['plqy'],
                                 load=load, init_mask=True, n_jobs=1)

    combined_data = []
    for i, data_tuple in enumerate(dataset):
        if len(data_tuple) == 3:
            smiles, graph, label = data_tuple
            mask = None
        else:
            smiles, graph, label, mask = data_tuple
        fp = torch.tensor(fp_data[i], dtype=torch.float32)
        combined_data.append((graph, fp, label, mask))
    return combined_data


def load_fingerprints(fp_file):
    df = pd.read_csv(fp_file)
    return torch.tensor(df.values, dtype=torch.float32)

train_data = pd.read_csv('./data/train_plqy.csv')
valid_data = pd.read_csv('./data/valid_plqy.csv')

scaler = StandardScaler()
train_data[['plqy']] = scaler.fit_transform(train_data[['plqy']])
valid_data[['plqy']] = scaler.transform(valid_data[['plqy']])

train_fp_solvent = load_fingerprints('./data/train_sol_plqy.csv')
valid_fp_solvent = load_fingerprints('./data/valid_sol_plqy.csv')
train_fp_smiles = load_fingerprints('./data/train_smiles_plqy.csv')
valid_fp_smiles = load_fingerprints('./data/valid_smiles_plqy.csv')

train_fp_extra = torch.tensor(train_data.iloc[:, 8:152].values, dtype=torch.float32)
valid_fp_extra = torch.tensor(valid_data.iloc[:, 8:152].values, dtype=torch.float32)
scaler_num = MinMaxScaler()
train_num = train_fp_extra[:, :8].numpy()
valid_num = valid_fp_extra[:, :8].numpy()
train_rest = train_fp_extra[:, 8:]
valid_rest = valid_fp_extra[:, 8:]
train_num_scaled = scaler_num.fit_transform(train_num)
valid_num_scaled = scaler_num.transform(valid_num)
train_fp_extra = torch.cat([torch.tensor(train_num_scaled, dtype=torch.float32), train_rest], dim=1)
valid_fp_extra = torch.cat([torch.tensor(valid_num_scaled, dtype=torch.float32), valid_rest], dim=1)
train_fp = torch.cat([train_fp_solvent, train_fp_smiles, train_fp_extra], dim=1)
valid_fp = torch.cat([valid_fp_solvent, valid_fp_smiles, valid_fp_extra], dim=1)


lds_weights = compute_lds_weights(train_data[['plqy']].values.flatten())

class GraphFingerprintsModel(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, fp_size, 
                 graph_feat_size=graph_feat_size, num_layers=num_layers, num_timesteps=num_timesteps, 
                 n_tasks=n_tasks, dropout=dropout):
        super(GraphFingerprintsModel, self).__init__()
        self.gnn = AttentiveFPGNN(node_feat_size=node_feat_size,
                                  edge_feat_size=edge_feat_size,
                                  num_layers=num_layers,
                                  graph_feat_size=graph_feat_size,
                                  dropout=dropout)
        self.readout = AttentiveFPReadout(feat_size=graph_feat_size,
                                          num_timesteps=num_timesteps,
                                          dropout=dropout)
        self.fp_fc = nn.Sequential(
            nn.Linear(fp_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, graph_feat_size)
        )
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, n_tasks)
        )

    def forward(self, g, node_feats, edge_feats, fingerprints):
        if edge_feats is None or 'he' not in g.edata.keys():
            num_edges = g.number_of_edges()
            edge_feats = torch.zeros((num_edges, e_feats)).to(g.device)
        node_feats = self.gnn(g, node_feats, edge_feats)
        graph_feats = self.readout(g, node_feats)
        fp_feats = self.fp_fc(fingerprints)
        combined_feats = torch.cat([graph_feats, fp_feats], dim=1)
        return self.predict(combined_feats)

class MolecularDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    if len(batch[0]) == 5:
        graphs, fps, labels, masks, weights = zip(*batch)
        weights = torch.stack(weights)
    else:
        graphs, fps, labels, masks = zip(*batch)
        weights = None
    graphs = dgl.batch(graphs)
    fps = torch.stack(fps)
    labels = torch.stack(labels)
    masks = torch.stack(masks) if masks[0] is not None else None
    return graphs, fps, labels, masks, weights


# === 读取 target 数据 ===
target_data = pd.read_csv('./input/input.csv')
target_fp_solvent = load_fingerprints('./input/target_sol_morgan.csv')
target_fp_smiles = load_fingerprints('./input/target_smiles_morgan.csv')

# === 数值特征标准化 ===
target_fp_extra = torch.tensor(target_data.iloc[:, 8:152].values, dtype=torch.float32)
target_num = target_fp_extra[:, :8].numpy()
target_rest = target_fp_extra[:, 8:]

# 加载预训练的 scaler_num
target_num_scaled = scaler_num.transform(target_num)
target_fp_extra = torch.cat([torch.tensor(target_num_scaled, dtype=torch.float32), target_rest], dim=1)

# === 拼接最终指纹 ===
target_fp = torch.cat([target_fp_solvent, target_fp_smiles, target_fp_extra], dim=1)

# === 标签标准化（只为保持接口一致，其实预测时不需操作标签） ===
target_data[['plqy']] = scaler.transform(target_data[['plqy']])  # 注意：仅为构造 dataset，不影响预测结果

# === 构造 dataset 与 dataloader ===
target_datasets = load_data_with_fp(target_data, target_fp, 'target', True)
target_dataset = MolecularDataset(target_datasets)
target_loader = DataLoader(target_dataset, batch_size=batch_size, collate_fn=collate_fn)


fp_size = target_fp.shape[1]
# 初始化模型
model = GraphFingerprintsModel(node_feat_size=n_feats,
                               edge_feat_size=e_feats,
                               graph_feat_size=graph_feat_size,
                               num_layers=num_layers,
                               num_timesteps=num_timesteps,
                               fp_size=fp_size,
                               n_tasks=n_tasks,
                               dropout=dropout).to(device)

# 加载保存的模型参数
model.load_state_dict(torch.load('Model_plqy.pth', map_location=device))
model.eval()




# 将预测结果保存到 CSV 文件
def save_predictions(predictions, file_name):
    df = pd.DataFrame(predictions, columns=['plqy'])
    df.to_csv(file_name, index=False)


# === 模型预测 ===
target_predictions = predict(model, target_loader)
target_scale_predictions = reverse_standardization(target_predictions, scaler)

# === 保存预测结果 ===
save_predictions(target_scale_predictions, './result/target_predictions_plqy.csv')
print("🎯 Target predictions saved to target_predictions_plqy.csv")




#######################################################################
# 预测摩尔吸收系数

graph_feat_size = 256
n_tasks = 1
dropout = 0.3
alpha = 0.6
batch_size = 32
learning_rate = 1e-3
epochs = 3
patience = 20
num_layers = 3
num_timesteps = 1

random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)





atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='hv')
bond_featurizer = AttentiveFPBondFeaturizer(bond_data_field='he')
n_feats = atom_featurizer.feat_size('hv')
e_feats = bond_featurizer.feat_size('he')


def load_data_with_fp(data, fp_data, name, load):
    dataset = MoleculeCSVDataset(data,
                                 smiles_to_graph=smiles_to_bigraph,
                                 node_featurizer=atom_featurizer,
                                 edge_featurizer=bond_featurizer,
                                 smiles_column='smiles',
                                 cache_file_path=str(name)+'_dataset_k.bin',
                                 task_names=['k'],
                                 load=load, init_mask=True, n_jobs=1)

    combined_data = []
    for i, data_tuple in enumerate(dataset):
        if len(data_tuple) == 3:
            smiles, graph, label = data_tuple
            mask = None
        else:
            smiles, graph, label, mask = data_tuple
        fp = torch.tensor(fp_data[i], dtype=torch.float32)
        combined_data.append((graph, fp, label, mask))
    return combined_data


def load_fingerprints(fp_file):
    df = pd.read_csv(fp_file)
    return torch.tensor(df.values, dtype=torch.float32)

train_data = pd.read_csv('./data/train_k.csv')
valid_data = pd.read_csv('./data/valid_k.csv')

scaler = StandardScaler()
train_data[['k']] = scaler.fit_transform(train_data[['k']])
valid_data[['k']] = scaler.transform(valid_data[['k']])

train_fp_solvent = load_fingerprints('./data/train_sol_k.csv')
valid_fp_solvent = load_fingerprints('./data/valid_sol_k.csv')
train_fp_smiles = load_fingerprints('./data/train_smiles_k.csv')
valid_fp_smiles = load_fingerprints('./data/valid_smiles_k.csv')

train_fp_extra = torch.tensor(train_data.iloc[:, 8:152].values, dtype=torch.float32)
valid_fp_extra = torch.tensor(valid_data.iloc[:, 8:152].values, dtype=torch.float32)
scaler_num = MinMaxScaler()
train_num = train_fp_extra[:, :8].numpy()
valid_num = valid_fp_extra[:, :8].numpy()
train_rest = train_fp_extra[:, 8:]
valid_rest = valid_fp_extra[:, 8:]
train_num_scaled = scaler_num.fit_transform(train_num)
valid_num_scaled = scaler_num.transform(valid_num)
train_fp_extra = torch.cat([torch.tensor(train_num_scaled, dtype=torch.float32), train_rest], dim=1)
valid_fp_extra = torch.cat([torch.tensor(valid_num_scaled, dtype=torch.float32), valid_rest], dim=1)
train_fp = torch.cat([train_fp_solvent, train_fp_smiles, train_fp_extra], dim=1)
valid_fp = torch.cat([valid_fp_solvent, valid_fp_smiles, valid_fp_extra], dim=1)


lds_weights = compute_lds_weights(train_data[['k']].values.flatten())



class GraphFingerprintsModel(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, fp_size, 
                 graph_feat_size=graph_feat_size, num_layers=num_layers, num_timesteps=num_timesteps, 
                 n_tasks=n_tasks, dropout=dropout):
        super(GraphFingerprintsModel, self).__init__()
        self.gnn = AttentiveFPGNN(node_feat_size=node_feat_size,
                                  edge_feat_size=edge_feat_size,
                                  num_layers=num_layers,
                                  graph_feat_size=graph_feat_size,
                                  dropout=dropout)
        self.readout = AttentiveFPReadout(feat_size=graph_feat_size,
                                          num_timesteps=num_timesteps,
                                          dropout=dropout)
        self.fp_fc = nn.Sequential(
            nn.Linear(fp_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, graph_feat_size)
        )
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, n_tasks)
        )

    def forward(self, g, node_feats, edge_feats, fingerprints):
        if edge_feats is None or 'he' not in g.edata.keys():
            num_edges = g.number_of_edges()
            edge_feats = torch.zeros((num_edges, e_feats)).to(g.device)
        node_feats = self.gnn(g, node_feats, edge_feats)
        graph_feats = self.readout(g, node_feats)
        fp_feats = self.fp_fc(fingerprints)
        combined_feats = torch.cat([graph_feats, fp_feats], dim=1)
        return self.predict(combined_feats)

class MolecularDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    if len(batch[0]) == 5:
        graphs, fps, labels, masks, weights = zip(*batch)
        weights = torch.stack(weights)
    else:
        graphs, fps, labels, masks = zip(*batch)
        weights = None
    graphs = dgl.batch(graphs)
    fps = torch.stack(fps)
    labels = torch.stack(labels)
    masks = torch.stack(masks) if masks[0] is not None else None
    return graphs, fps, labels, masks, weights



# === 读取 target 数据 ===
target_data = pd.read_csv('./input/input.csv')
target_fp_solvent = load_fingerprints('./input/target_sol_morgan.csv')
target_fp_smiles = load_fingerprints('./input/target_smiles_morgan.csv')

# === 数值特征标准化 ===
target_fp_extra = torch.tensor(target_data.iloc[:, 8:152].values, dtype=torch.float32)
target_num = target_fp_extra[:, :8].numpy()
target_rest = target_fp_extra[:, 8:]

# 加载预训练的 scaler_num
target_num_scaled = scaler_num.transform(target_num)
target_fp_extra = torch.cat([torch.tensor(target_num_scaled, dtype=torch.float32), target_rest], dim=1)

# === 拼接最终指纹 ===
target_fp = torch.cat([target_fp_solvent, target_fp_smiles, target_fp_extra], dim=1)

# === 标签标准化（只为保持接口一致，其实预测时不需操作标签） ===
target_data[['abs']] = scaler.transform(target_data[['abs']])  # 注意：仅为构造 dataset，不影响预测结果

# === 构造 dataset 与 dataloader ===
target_datasets = load_data_with_fp(target_data, target_fp, 'target', True)
target_dataset = MolecularDataset(target_datasets)
target_loader = DataLoader(target_dataset, batch_size=batch_size, collate_fn=collate_fn)



fp_size = target_fp.shape[1]
# 初始化模型
model = GraphFingerprintsModel(node_feat_size=n_feats,
                               edge_feat_size=e_feats,
                               graph_feat_size=graph_feat_size,
                               num_layers=num_layers,
                               num_timesteps=num_timesteps,
                               fp_size=fp_size,
                               n_tasks=n_tasks,
                               dropout=dropout).to(device)

# 加载保存的模型参数
model.load_state_dict(torch.load('Model_k.pth', map_location=device))
model.eval()




# 将预测结果保存到 CSV 文件
def save_predictions(predictions, file_name):
    df = pd.DataFrame(predictions, columns=['k'])
    df.to_csv(file_name, index=False)


# === 模型预测 ===
target_predictions = predict(model, target_loader)
target_scale_predictions = reverse_standardization(target_predictions, scaler)

# === 保存预测结果 ===
save_predictions(target_scale_predictions, './result/target_predictions_k.csv')
print("🎯 Target predictions saved to target_predictions_k.csv")