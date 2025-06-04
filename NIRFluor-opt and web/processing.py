import pandas as pd
from rdkit import Chem
from rdkit.Chem import (MolToSmiles, rdMMPA, MACCSkeys, AllChem, rdMolDescriptors, 
                        DataStructs, MolFromSmiles, CombineMols, Draw)
from rdkit.Chem import Descriptors
from rdkit.Contrib.SA_Score import sascorer
from tqdm import tqdm
import csv
import os
import re
import random
import numpy as np
import joblib
from typing import List, Dict, Optional, Tuple
import glob
from PIL import Image, ImageDraw, ImageFont


def process(similarity_value):

    image_folder = './results/molecule_images'
    png_files = glob.glob(os.path.join(image_folder, '*.png'))

    for file_path in png_files:
        try:
            os.remove(file_path)
            print(f"已删除 PNG 文件: {file_path}")
        except Exception as e:
            print(f"删除 PNG 失败: {file_path}，原因: {e}")


    image_folder = './results/rules_images'
    png_files = glob.glob(os.path.join(image_folder, '*.png'))

    for file_path in png_files:
        try:
            os.remove(file_path)
            print(f"已删除 PNG 文件: {file_path}")
        except Exception as e:
            print(f"删除 PNG 失败: {file_path}，原因: {e}")
            

    results_folder = './results'
    csv_files = glob.glob(os.path.join(results_folder, '*.csv'))

    for file_path in csv_files:
        try:
            os.remove(file_path)
            print(f"已删除 CSV 文件: {file_path}")
        except Exception as e:
            print(f"删除 CSV 失败: {file_path}，原因: {e}")


    ##############################################  1_目标分子拆分 ##############################################
    
    similarity_value = similarity_value

    def fragment_molecules(molecules): # 片段化分子
        fragments = []
        for mol in tqdm(molecules, desc="Fragmenting molecules", ncols=100):
            if mol is not None:
                fragment = rdMMPA.FragmentMol(mol, minCuts=min_cuts, maxCuts=max_cuts, maxCutBonds=max_cut_bonds, resultsAsMols=asmol)
                fragments.append(fragment)
        return fragments

    def frag(fragments, original_molecules): # 处理分子碎片的函数
        mol_nums = 0
        results = []
        for mols, original_mol in zip(fragments, original_molecules):
            mol_nums += 1
            cut_nums = 0
            for cuts in mols:
                cut_nums += 1
                scaffold = cuts[0]
                frag = cuts[1]
                if scaffold is not None:
                    scaffold_smiles = MolToSmiles(scaffold)
                else:
                    scaffold_smiles = None
                if frag is not None:
                    frag_smiles = MolToSmiles(frag)
                else:
                    frag_smiles = None
                if original_mol is not None:
                    original_smiles = MolToSmiles(original_mol)
                else:
                    original_smiles = None
                results.append([original_smiles, cut_nums, scaffold_smiles, frag_smiles])
        return results

    data_target = pd.read_csv('./input/target_m.csv') 

    try:
        first_smiles = data_target['smiles'].iloc[0]
        mol = Chem.MolFromSmiles(first_smiles)
        if mol is not None:
            abs_molecules = [mol]
        else:
            raise RuntimeError(f"输入分子无效，运行结束，请检查 SMILES：{first_smiles}")

        cut_nums = [1, 2, 3]
        fragments_df = pd.DataFrame()

        for cut_num in cut_nums:
            min_cuts = cut_num
            max_cuts = cut_num
            max_cut_bonds = 100
            asmol = True

            abs_fragments = fragment_molecules(abs_molecules)
            fragments = abs_fragments
            columns = ['smiles', 'cutnum', 'scaffold', 'fragment']
            results = []

            results = frag(fragments, abs_molecules)
            df = pd.DataFrame(results, columns=columns)

            df['combined'] = df.apply(
                lambda row: f"{row.iloc[2]}.{row.iloc[3]}" if pd.notna(row.iloc[2]) and pd.notna(row.iloc[3]) else (row.iloc[2] if pd.notna(row.iloc[2]) else row.iloc[3]),
                axis=1
            )
            
            fragments_df = pd.concat([fragments_df, df], ignore_index=True)

        # 保存结果到文件
        fragments_df.to_csv('./results/target_fragment.csv', index=False)

        if fragments_df.empty:
            raise RuntimeError(f"该分子无法被有效拆分：{first_smiles}")

    except Exception as e:
        raise RuntimeError("无法拆分，跳过")
        


    ##############################################  2_规则寻找 ##############################################

    def tanimoto_similarity(fp1, fp2): # 计算Tanimoto相似性
        return DataStructs.TanimotoSimilarity(fp1, fp2)

    first_smiles = data_target.loc[0, 'smiles']
    mol = Chem.MolFromSmiles(first_smiles)

    # 生成 MACCS 指纹
    if mol is not None:
        fp = MACCSkeys.GenMACCSKeys(mol)
        fp_list = list(fp)
        maccs_df = pd.DataFrame([fp_list], columns=[f'bit_{i}' for i in range(167)])
        target_maccs_df = pd.concat([data_target.loc[[0]], maccs_df], axis=1)

    ########### 规则过滤 ###########
    rules_maccs = pd.read_csv('./data/转换规则_MACCS.csv')

    smiles1 = target_maccs_df.iloc[0, 0]  # 第一行第一列为smiles
    fingerprint1 = target_maccs_df.iloc[0, 1:].values.astype(int)  # 第一行其余列为分子指纹

    mol1 = Chem.MolFromSmiles(smiles1)
    fp1 = DataStructs.CreateFromBitString(''.join(fingerprint1.astype(str)))

    output_rows = []

    for _, row in rules_maccs.iterrows():
        smiles2 = row[2]  # 第三列为smiles
        fingerprint2 = row[3:].values.astype(int)  # 第四列到最后一列为分子指纹
        
        # 转换为RDKit的分子指纹对象
        fp2 = DataStructs.CreateFromBitString(''.join(fingerprint2.astype(str)))
        
        # 计算相似性
        similarity = tanimoto_similarity(fp1, fp2)
        
        # 如果相似性大于阈值，将该行保存
        if similarity > similarity_value:
            output_rows.append(row)

    target_similary_rules_df = pd.DataFrame(output_rows)
    target_similary_rules_df.to_csv('./results/target_similary_rules.csv', index=False)

    if target_similary_rules_df.empty:
        raise RuntimeError("没有找到匹配的规则，运行结束")

    print('----------完成目标分子规则筛选，生成文件：target_similary_rules.csv----------')


    ########### 文件优化 ###########
    element_tran_unique = target_similary_rules_df[['element_tran']] # 仅保留 element_tran 这一列
    element_tran_unique = element_tran_unique.drop_duplicates() # 去重

    if 'element_tran' not in element_tran_unique.columns: # 检查 element_tran 列是否存在
        raise ValueError("target_similary_rules.csv 文件中没有找到 'element_tran' 列，请检查文件内容！")

    element_tran_unique[['node1', 'node2']] = element_tran_unique['element_tran'].str.split(' --->>>--- ', expand=True) # 使用 str.split() 方法拆分 element_tran 列

    new_rows = []
    for index, row in element_tran_unique.iterrows(): # 遍历每一行
        for i in range(1, 4): # 对每一行生成三行，分别替换 [*:*] 为 [*:1]、[*:2]、[*:3]
            new_row = row.copy()  # 复制当前行
            new_row = new_row.replace(r'\[\*:\*\]', f'[*:{i}]', regex=True)  # 替换 [*:*] 为 [*:i]
            new_rows.append(new_row)  # 将新行添加到列表中

    # 将新行列表转换为新的 DataFrame
    target_rules_df = pd.DataFrame(new_rows)
    target_rules_df.to_csv('./results/target_rules.csv', index=False)


    # 将使用H进行替换的规则单独分开
    file_path = './results/target_rules.csv'
    df = pd.read_csv(file_path)

    # 检查 node1 列是否为空，并提取这些行
    empty_node1_rows = df[df['node1'].isna()]

    # 如果没有找到空值行，提示用户
    if empty_node1_rows.empty:
        print('----------完成目标分子规则整理，生成文件：target_rules.csv----------')
    else:
        # 筛选出每隔3行的数据（索引从0开始，因此选择第0、3、6...行）
        filtered_rows = empty_node1_rows.iloc[::3]

        # 保存筛选后的结果到新的 CSV 文件
        output_file = './results/target_rules_replace.csv'
        filtered_rows.to_csv(output_file, index=False)
        print(f'----------完成目标分子规则整理，生成文件：target_rules.csv 和 target_rules_replace.csv----------')


    ##############################################  4_改造部位确定 ##############################################
    df1 = fragments_df
    file2_path = './results/target_rules.csv'
    df2 = pd.read_csv(file2_path)

    # 创建一个字典，将第二份文件的 node1 映射到 node2
    node1_to_node2 = dict(zip(df2['node1'], df2['node2']))

    # 创建一个集合来记录被使用的键
    used_keys = set()

    # 定义一个函数，用于替换 combined 列中的 node1 部分为 node2
    def replace_node1_with_node2(combined_value):
        if pd.isna(combined_value):
            return combined_value  # 如果 combined 是空值，直接返回
        nodes = str(combined_value).split('.')  # 确保 combined_value 是字符串类型
        replaced_nodes = []
        for node in nodes:
            if node in node1_to_node2:
                used_keys.add(node)  # 记录被使用的键
                replaced_nodes.append(str(node1_to_node2[node]))
            else:
                replaced_nodes.append(node)
        return '.'.join(replaced_nodes)  # 重新组合为字符串

    # 检查 combined 列是否包含第二份文件的 node1 列的值，并对匹配的行进行替换操作
    matched_rows = df1[df1['combined'].apply(lambda x: any(node in str(x).split('.') for node in node1_to_node2.keys()))].copy()

    # 如果有匹配的行，进行替换操作
    if not matched_rows.empty:
        matched_rows['combined'] = matched_rows['combined'].apply(replace_node1_with_node2)

    # 如果没有匹配的行，提示用户
    if matched_rows.empty:
        raise RuntimeError("没有找到匹配的行，运行结束")
    else:
        # 保存结果到新的 CSV 文件
        output_file = './results/new_m_replace.csv'
        matched_rows.to_csv(output_file, index=False)
        
        # 提取被使用的键值对并保存
        used_pairs = df2[df2['node1'].isin(used_keys)]
        used_pairs_file = './results/used_mapping_pairs.csv'
        used_pairs.to_csv(used_pairs_file, index=False)
        print(f"--------已保存被使用的键值对到 {used_pairs_file}--------")


    ################# new_m_replace.csv文件整理 #################
    file_path = './results/new_m_replace.csv'
    df = pd.read_csv(file_path)

    # 检查文件中是否包含 'combined' 列
    if 'combined' not in df.columns:
        raise RuntimeError("文件无'combined' 列")


    # 仅保留 'combined' 列
    combined_df = df[['combined']]
    df = combined_df


    # 拆分`combined`列
    split_columns = df['combined'].str.split('\.', expand=True)

    # 为拆分后的列命名
    split_columns.columns = [f'element_{i+1}' for i in range(split_columns.shape[1])]

    # 将拆分后的列合并回原DataFrame
    df = pd.concat([df, split_columns], axis=1)

    # 删除原始的`combined`列
    df.drop(columns=['combined'], inplace=True)


    # 定义一个函数来检查并分配元素到不同列
    def reassign_columns(row):
        # 获取当前行的第二、三、四列的值
        col2, col3, col4 = row[1], row[2], row[3]
        
        # 确保每个值都是字符串类型，并处理空值
        col2 = str(col2) if pd.notna(col2) else ''
        col3 = str(col3) if pd.notna(col3) else ''
        col4 = str(col4) if pd.notna(col4) else ''
        
        # 初始化新的列
        new_col2, new_col3, new_col4 = [], [], []
        
        # 对每一列进行检查
        if "[*:1]" in col2:
            new_col2.append(col2)
        elif "[*:2]" in col2:
            new_col3.append(col2)
        elif "[*:3]" in col2:
            new_col4.append(col2)
        
        if "[*:1]" in col3:
            new_col2.append(col3)
        elif "[*:2]" in col3:
            new_col3.append(col3)
        elif "[*:3]" in col3:
            new_col4.append(col3)
        
        if "[*:1]" in col4:
            new_col2.append(col4)
        elif "[*:2]" in col4:
            new_col3.append(col4)
        elif "[*:3]" in col4:
            new_col4.append(col4)
        
        return ','.join(new_col2), ','.join(new_col3), ','.join(new_col4)

    if df.shape[1] != 4:
        raise RuntimeError(f"警告：数据框的列数不是 4，而是 {df.shape[1]} 列，请检查数据。")

    # 处理每一行并重新分配到新的列
    for i, row in df.iterrows():
        df.iloc[i, 1], df.iloc[i, 2], df.iloc[i, 3] = reassign_columns(row)

    df.to_csv('./results/new_m_replace.csv', index=False)


    ################# new_m_replace.csv文件进一步整理 #################
    ################# 以下内容为应对无环境生成而加入的，对正常情况也适用 #################
    input_file = "./results/new_m_replace.csv"
    output_file = "./results/new_m_replace.csv"

    df = pd.read_csv(input_file)

    # 检查每行的元素数量，删除少于 2 个元素的行
    df = df.dropna(thresh=2)
    df.to_csv(output_file, index=False)


    ################# 数据补充
    input_file = "./results/new_m_replace.csv"
    output_file = "./results/new_m_replace.csv"

    df = pd.read_csv(input_file)

    # 定义处理函数
    def process_row(row):
        element_1 = row["element_1"]
        element_2 = row["element_2"]
        element_3 = row["element_3"]
        element_4 = row["element_4"]

        # 检查 element_1 中的标记
        has_marker_1 = "[*:1]" in element_1
        has_marker_2 = "[*:2]" in element_1
        has_marker_3 = "[*:3]" in element_1

        # 情况 1: element_1 有 [*:1] 和 [*:2]，但没有 [*:3]
        if has_marker_1 and has_marker_2 and not has_marker_3:
            if pd.isna(element_2) and pd.isna(element_3):
                # 如果 element_2 和 element_3 都为空，无法填充，跳过
                pass
            elif pd.isna(element_2):
                # 如果 element_2 为空，将 element_3 复制给 element_2
                row["element_2"] = element_3
            elif pd.isna(element_3):
                # 如果 element_3 为空，将 element_2 复制给 element_3
                row["element_3"] = element_2

        # 情况 2: element_1 有 [*:1]、[*:2] 和 [*:3]
        elif has_marker_1 and has_marker_2 and has_marker_3:
            # 统计 element_2、element_3、element_4 中的空值数量
            null_count = sum(pd.isna([element_2, element_3, element_4]))

            if null_count == 2:
                # 如果有两列为空，将非空的一列复制给另外两列
                non_null_value = next((x for x in [element_2, element_3, element_4] if not pd.isna(x)), None)
                if non_null_value is not None:
                    row["element_2"] = non_null_value
                    row["element_3"] = non_null_value
                    row["element_4"] = non_null_value
            elif null_count == 1:
                # 如果只有一列为空，随机选择一列复制给空的一列
                non_null_columns = [col for col in ["element_2", "element_3", "element_4"] if not pd.isna(row[col])]
                if non_null_columns:
                    random_column = random.choice(non_null_columns)
                    for col in ["element_2", "element_3", "element_4"]:
                        if pd.isna(row[col]):
                            row[col] = row[random_column]

        return row

    # 处理每一行
    df = df.apply(process_row, axis=1)
    df.to_csv(output_file, index=False)

    ### 标签整理 ###
    file_path = './results/new_m_replace.csv'
    df = pd.read_csv(file_path)

    # 定义一个函数用于替换指定列中的[*:1], [*:2], [*:3]
    def replace_elements_in_column(column, replacement):
        return column.str.replace(r'\[\*:1\]|\[\*:2\]|\[\*:3\]', replacement, regex=True)

    # 替换第二列、第三列和第四列的元素
    df.iloc[:, 1] = replace_elements_in_column(df.iloc[:, 1], '[*:1]')
    df.iloc[:, 2] = replace_elements_in_column(df.iloc[:, 2], '[*:2]')
    df.iloc[:, 3] = replace_elements_in_column(df.iloc[:, 3], '[*:3]')

    output_file = './results/new_m_replace.csv'  # 替换为输出文件的路径
    df.to_csv(output_file, index=False)

    print('----------完成改造部位对应，生成文件：new_m_replace.csv----------')

    ##############################################  5_生成新分子 ##############################################


    def find_connection_atom(mol, map_num):
        """找到具有指定映射编号的原子索引"""
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() == map_num:
                return atom.GetIdx()
        return None

    # 读取CSV文件
    df = pd.read_csv('./results/new_m_replace.csv')
    unique_molecules = set()

    for i, row in df.iterrows():
        try:
            # 获取非空元素并存储到列表
            non_empty_elements = row.dropna().tolist()
            num_elements = len(non_empty_elements)

            if num_elements == 2:
                smiles_part1 = non_empty_elements[0]
                smiles_part2 = non_empty_elements[1]
                mol_part1 = Chem.MolFromSmiles(smiles_part1)
                mol_part2 = Chem.MolFromSmiles(smiles_part2)

                if None in (mol_part1, mol_part2):
                    raise ValueError("SMILES 解析失败")

                atom_to_connect_1 = find_connection_atom(mol_part1, 1)
                atom_to_connect_2 = find_connection_atom(mol_part2, 1)

                if None in (atom_to_connect_1, atom_to_connect_2):
                    raise ValueError("无法找到所有连接点")

                combined_mol = Chem.RWMol(Chem.CombineMols(mol_part1, mol_part2))
                offset = mol_part1.GetNumAtoms()
                combined_mol.AddBond(atom_to_connect_1, atom_to_connect_2 + offset, Chem.BondType.SINGLE)

                # 清除映射编号并生成规范化SMILES
                for atom in combined_mol.GetAtoms():
                    atom.SetAtomMapNum(0)
                final_mol = combined_mol.GetMol()
                final_smiles = Chem.MolToSmiles(final_mol, canonical=True)
                final_smiles = final_smiles.replace('**', '')
                
                # 实时去重
                if final_smiles not in unique_molecules:
                    unique_molecules.add(final_smiles)

            elif num_elements == 3:
                smiles_part1 = non_empty_elements[0]
                smiles_part2 = non_empty_elements[1]
                smiles_part3 = non_empty_elements[2]
                mol_part1 = Chem.MolFromSmiles(smiles_part1)
                mol_part2 = Chem.MolFromSmiles(smiles_part2)
                mol_part3 = Chem.MolFromSmiles(smiles_part3)

                if None in (mol_part1, mol_part2, mol_part3):
                    raise ValueError("SMILES 解析失败")

                atom_to_connect_1 = find_connection_atom(mol_part1, 1)
                atom_to_connect_2 = find_connection_atom(mol_part1, 2)
                atom_to_connect_3 = find_connection_atom(mol_part2, 1)
                atom_to_connect_4 = find_connection_atom(mol_part3, 2)

                if None in (atom_to_connect_1, atom_to_connect_2, atom_to_connect_3, atom_to_connect_4):
                    raise ValueError("无法找到所有连接点")

                combined_mol = Chem.RWMol(Chem.CombineMols(mol_part1, mol_part2))
                combined_mol = Chem.RWMol(Chem.CombineMols(combined_mol, mol_part3))
                offset_2 = mol_part1.GetNumAtoms()
                offset_3 = offset_2 + mol_part2.GetNumAtoms()

                combined_mol.AddBond(atom_to_connect_1, atom_to_connect_3 + offset_2, Chem.BondType.SINGLE)
                combined_mol.AddBond(atom_to_connect_2, atom_to_connect_4 + offset_3, Chem.BondType.SINGLE)

                # 清除映射编号并生成规范化SMILES
                for atom in combined_mol.GetAtoms():
                    atom.SetAtomMapNum(0)
                final_mol = combined_mol.GetMol()
                final_smiles = Chem.MolToSmiles(final_mol, canonical=True)
                final_smiles = final_smiles.replace('**', '')
                
                if final_smiles not in unique_molecules:
                    unique_molecules.add(final_smiles)

            elif num_elements == 4:
                smiles_part1 = non_empty_elements[0]
                smiles_part2 = non_empty_elements[1]
                smiles_part3 = non_empty_elements[2]
                smiles_part4 = non_empty_elements[3]
                mol_part1 = Chem.MolFromSmiles(smiles_part1)
                mol_part2 = Chem.MolFromSmiles(smiles_part2)
                mol_part3 = Chem.MolFromSmiles(smiles_part3)
                mol_part4 = Chem.MolFromSmiles(smiles_part4)

                if None in (mol_part1, mol_part2, mol_part3, mol_part4):
                    raise ValueError("SMILES 解析失败")

                atom_to_connect_1 = find_connection_atom(mol_part1, 1)
                atom_to_connect_2 = find_connection_atom(mol_part1, 2)
                atom_to_connect_3 = find_connection_atom(mol_part1, 3)
                atom_to_connect_4 = find_connection_atom(mol_part2, 1)
                atom_to_connect_5 = find_connection_atom(mol_part3, 2)
                atom_to_connect_6 = find_connection_atom(mol_part4, 3)

                if None in (atom_to_connect_1, atom_to_connect_2, atom_to_connect_3, 
                        atom_to_connect_4, atom_to_connect_5, atom_to_connect_6):
                    raise ValueError("无法找到所有连接点")

                combined_mol = Chem.RWMol(Chem.CombineMols(mol_part1, mol_part2))
                combined_mol = Chem.RWMol(Chem.CombineMols(combined_mol, mol_part3))
                combined_mol = Chem.RWMol(Chem.CombineMols(combined_mol, mol_part4))

                offset_2 = mol_part1.GetNumAtoms()
                offset_3 = offset_2 + mol_part2.GetNumAtoms()
                offset_4 = offset_3 + mol_part3.GetNumAtoms()

                combined_mol.AddBond(atom_to_connect_1, atom_to_connect_4 + offset_2, Chem.BondType.SINGLE)
                combined_mol.AddBond(atom_to_connect_2, atom_to_connect_5 + offset_3, Chem.BondType.SINGLE)
                combined_mol.AddBond(atom_to_connect_3, atom_to_connect_6 + offset_4, Chem.BondType.SINGLE)

                # 清除映射编号并生成规范化SMILES
                for atom in combined_mol.GetAtoms():
                    atom.SetAtomMapNum(0)
                final_mol = combined_mol.GetMol()
                final_smiles = Chem.MolToSmiles(final_mol, canonical=True)
                final_smiles = final_smiles.replace('**', '')
                
                if final_smiles not in unique_molecules:
                    unique_molecules.add(final_smiles)

        except Exception as e:
            raise RuntimeError("警告")

    df_final = pd.DataFrame(list(unique_molecules), columns=['smiles'])
    df_final = df_final.drop_duplicates()

    output_path = './results/基团替换.csv'
    df_final.to_csv(output_path, index=False)

    print('----------完成分子生成，生成文件：基团替换.csv----------')


    ##############################################  6_生成 H 取代分子 ##############################################
    new_m_finally_H = []
    smiles = data_target['smiles'].iloc[0]
    file_path2 = './results/target_rules_replace.csv'

    if not os.path.exists(file_path2):
        raise RuntimeError("文件不存在，无H原子替换规则")

    df2 = pd.read_csv(file_path2)
    substitute_smiles_list = df2['node2'].dropna().tolist()

    # 存储规范化后的SMILES用于去重
    canonical_smiles_set = set()

    for substitute_smiles in substitute_smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        substitute_mol = Chem.MolFromSmiles(substitute_smiles)

        replaceable_atoms = []
        for atom in mol.GetAtoms():
            if atom.GetTotalNumHs() > 0:
                replaceable_atoms.append(atom.GetIdx())

        if len(replaceable_atoms) == 0:
            raise RuntimeError("警告")

        replace_idx = None
        for atom in substitute_mol.GetAtoms():
            if atom.GetSymbol() == "*":
                replace_idx = atom.GetIdx()
                break

        if replace_idx is None:
            raise ValueError("没有找到[*:1]标记位置")

        combined_mol = Chem.CombineMols(mol, substitute_mol)

        for chosen_atom_idx in replaceable_atoms:
            mol_copy = Chem.RWMol(combined_mol)
            substitute_atom = mol_copy.GetAtomWithIdx(len(mol.GetAtoms()) + replace_idx)
            mol_copy.AddBond(chosen_atom_idx, substitute_atom.GetIdx(), Chem.BondType.SINGLE)
            
            try:
                final_mol = mol_copy.GetMol()
                # 规范化SMILES表示
                final_smiles = Chem.MolToSmiles(final_mol, canonical=True)
                final_smiles = final_smiles.replace('[*:1]', '')
                
                # 检查是否已存在（规范化后的SMILES）
                if final_smiles not in canonical_smiles_set:
                    canonical_smiles_set.add(final_smiles)
                    new_m_finally_H.append(final_smiles)
            except:
                raise RuntimeError("警告")

    unique_smiles = list(set(new_m_finally_H))
    data_target = pd.DataFrame(unique_smiles, columns=['smiles'])
    output_file_path = './results/H替换.csv'
    data_target.to_csv(output_file_path, index=False)
    print('----------完成H原子替换，生成文件：H替换.csv----------')

    ### 文件合并 ###
    file1 = './results/基团替换.csv'
    file2 = './results/H替换.csv'
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    merged_df = pd.concat([df1, df2], ignore_index=True)
    merged_df.to_csv('./results/merged_file.csv', index=False)
    print("两个CSV文件已成功合并并保存为 'merged_file.csv'")

    ##############################################  7_新分子预测 ##############################################

    # 1. 数据加载和预处理
    df = pd.read_csv('./results/merged_file.csv')
    total_molecules = len(df)
    print(f"成功加载 {total_molecules} 个分子")

    # 2. SMILES有效性检查
    print("\n正在检查SMILES有效性...")
    def is_valid_smiles(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False

    valid_mask = df['smiles'].apply(is_valid_smiles)
    invalid_count = (~valid_mask).sum()

    if invalid_count > 0:
        print(f"发现 {invalid_count} 个无效SMILES，将被移除")
        df = df[valid_mask].copy()
        print(f"剩余有效分子数: {len(df)}")
    else:
        print("所有SMILES都是有效的")

    # 3. 生成Morgan指纹
    print("\n开始生成Morgan指纹...")
    def get_morgan_fingerprint(smiles, radius=2, n_bits=2048):
        mol = Chem.MolFromSmiles(smiles)
        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return list(morgan_fp)

    fingerprints = []
    for smiles in tqdm(df['smiles'], desc="生成指纹", total=len(df)):
        fp = get_morgan_fingerprint(smiles)
        fingerprints.append(fp)

    print("\n正在转换指纹数据...")
    fingerprint_df = pd.DataFrame(fingerprints, columns=[f'Bit_{i+1}' for i in range(2048)])
    combined_df = pd.concat([df.reset_index(drop=True), fingerprint_df], axis=1)

    # 4. 模型预测
    print("\n正在加载模型并进行预测...")
    try:
        stacking_model = joblib.load("./data/stacking_model_full.pkl")
        X_new = combined_df.iloc[:, 1:].values
        y_new_prob = stacking_model.predict_proba(X_new)[:, 1]
        y_new_pred = stacking_model.predict(X_new)
        
        # 创建结果DataFrame
        result_df = pd.DataFrame({
            'smiles': combined_df['smiles'],
            'predicted_prob': y_new_prob,
            'predicted_label': y_new_pred
        }).sort_values(by='predicted_prob', ascending=False)
        
        label_counts = result_df['predicted_label'].value_counts()
        print("\n预测结果统计：")
        print(f"标签为 0 的数量: {label_counts.get(0, 0)}")
        print(f"标签为 1 的数量: {label_counts.get(1, 0)}")
        print(f"预测概率范围: {result_df['predicted_prob'].min():.3f} - {result_df['predicted_prob'].max():.3f}")
        
        result_df.to_csv('./results/merged_file_pred.csv', index=False)

    except Exception as e:
        print(f"\n模型加载或预测出错: {str(e)}")

    ################################################################### 展示使用的规则

    # 读取两个 CSV 文件
    file1_path = './results/used_mapping_pairs.csv'
    file2_path = './results/target_rules_replace.csv'

    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # 合并两个 DataFrame（垂直堆叠）
    combined_df = pd.concat([df1, df2], ignore_index=True)

    df = combined_df

    # 定义一个函数，将 [*:1], [*:2], ... 替换为 [*]
    def normalize_smiles(smiles):
        if pd.isna(smiles):
            return smiles
        return re.sub(r'\[\*:\d+\]', '[*:1]', smiles)

    # 对 element_tran, node1, node2 列进行标准化处理
    df['element_tran'] = df['element_tran'].apply(normalize_smiles)
    df['node1'] = df['node1'].apply(normalize_smiles)
    df['node2'] = df['node2'].apply(normalize_smiles)

    # 去重（基于所有列）
    df_unique = df.drop_duplicates()

    # 保存去重后的结果
    output_path1 = './results/transform_rules.csv'
    df_unique.to_csv(output_path1, index=False)

    ################################################################ 可合成性分析
    # SA Score值越小表示分子越容易合成(典型范围1-10，小于3表示容易合成)

    def calculate_sa_scores(input_csv, output_csv, smiles_column=0):
        with open(input_csv, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            
            # 读取标题行并添加新列
            headers = next(reader)
            headers.append('SA_Score')
            writer.writerow(headers)
                                                                                                                                                                                    
            for row in reader:
                smiles = row[smiles_column]
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        sa_score = sascorer.calculateScore(mol)
                        row.append(f"{sa_score:.2f}")  # 保留两位小数
                    else:
                        row.append('Invalid SMILES')
                except:
                    row.append('Calculation Error')
                
                writer.writerow(row)

    # 使用示例 - 替换为你的实际文件路径
    input_file = './results/merged_file_pred.csv'
    output_file = './results/new_molecules.csv'
    calculate_sa_scores(input_file, output_file)
    print("完成可合成性分析")