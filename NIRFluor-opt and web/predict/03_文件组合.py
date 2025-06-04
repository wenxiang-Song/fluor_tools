import pandas as pd
import os

# 输入文件路径
file1 = './predict/result/target_predictions_abs.csv'
file2 = './predict/result/target_predictions_em.csv'
file3 = './predict/result/target_predictions_plqy.csv'
file4 = './predict/result/target_predictions_k.csv'

# 读取每个文件（只有一列）
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)
df4 = pd.read_csv(file4)

# 合并所有列
combined_df = pd.concat([df1, df2, df3, df4], axis=1)

# 保存为新的文件
combined_df.to_csv('./predict/result/target_predictions.csv', index=False)

print("✅ 合并完成，已保存为 target_predictions.csv")


def delete_all_bin_files(folder_path):
    deleted_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.bin'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    deleted_files.append(file_path)
                    print(f"✅ 已删除：{file_path}")
                except Exception as e:
                    print(f"❌ 删除失败：{file_path}，原因：{e}")
    if not deleted_files:
        print("📂 未找到任何 .bin 文件")
    else:
        print(f"🧹 总共删除了 {len(deleted_files)} 个 .bin 文件")

# 直接执行
delete_all_bin_files('./predict/')  # 替换为实际路径
