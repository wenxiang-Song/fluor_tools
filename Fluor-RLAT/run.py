
import pandas as pd
import subprocess

# 文件路径
csv_file = './input/target.csv'  # 替换为你的实际路径

# 手动输入的新值（右侧名称）
new_smiles = 'CC(C)(C)c1ccc2ccc(C(c3ccccc3)c3ccccc3)c-2cc1'  # 示例 SMILES
new_solvent_name = 'MeCN'  # 示例溶剂名称（右侧）

# 溶剂名称到结构式的映射（右 -> 左）
solvent_mapping = {
    'CH2Cl2': 'ClCCl',
    'MeOH': 'CO',
    'EtOH': 'CCO',
    'CHCl3': 'ClC(Cl)Cl',
    'MeCN': 'CC#N',
    'THF': 'C1CCOC1',
    'Toluene': 'Cc1ccccc1',
    'DMSO': 'CS(C)=O',
    'H2O': 'O',
    'Benzene': 'c1ccccc1'
}

# 将输入的右侧溶剂名称转换为左侧结构式
if new_solvent_name not in solvent_mapping:
    raise ValueError(f"❌ 输入的溶剂名称 '{new_solvent_name}' 未在映射表中找到")
new_solvent = solvent_mapping[new_solvent_name]

# 读取原始 CSV 文件
df = pd.read_csv(csv_file)

# 替换第一行的 smiles 和 solvent
if 'smiles' in df.columns and 'solvent' in df.columns:
    df.at[0, 'smiles'] = new_smiles
    df.at[0, 'solvent'] = new_solvent
else:
    raise ValueError("❌ CSV 文件中未找到 'smiles' 或 'solvent' 列")

# 保存更新后的 CSV 文件
df.to_csv(csv_file, index=False)

print(f"✅ 第一行 'smiles' 和 'solvent' 已替换为：{new_smiles}, {new_solvent}，并保存至：{csv_file}")





print("🚀 正在运行 01_数据预处理.py...")
subprocess.run(['python', '01_数据预处理.py'], check=True)

print("🚀 正在运行 02_性质预测.py...")
subprocess.run(['python', '02_性质预测.py'], check=True)

print("🚀 正在运行 03_文件组合.py...")
subprocess.run(['python', '03_文件组合.py'], check=True)