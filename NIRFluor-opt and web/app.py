from flask import Flask, render_template, request, send_file
import os
import processing
import pandas as pd
import subprocess

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './input'
app.config['RESULT_FOLDER'] = './results'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Home 页面
@app.route('/')
def home():
    return render_template('home.html')

# Optimization 页面
@app.route('/index')
def optimization():
    return render_template('index.html')

# Prediction 页面
@app.route('/prediction', methods=['GET', 'POST'])
def predict():
    print("📥 正在进入 /prediction 路由处理")
    if request.method == 'POST':
        try:
            new_smiles = request.form['smiles']
            new_solvent_name = request.form['solvent']

            # 溶剂映射
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

            if new_solvent_name not in solvent_mapping:
                return render_template('prediction.html', error='溶剂名不在支持列表中')

            new_solvent = solvent_mapping[new_solvent_name]

            df = pd.read_csv('./predict/input/target.csv')
            df.at[0, 'smiles'] = new_smiles
            df.at[0, 'solvent'] = new_solvent
            df.to_csv('./predict/input/target.csv', index=False)

            subprocess.run(['python', './predict/01_数据预处理.py'], check=True)
            subprocess.run(['python', './predict/02_性质预测.py'], check=True)
            subprocess.run(['python', './predict/03_文件组合.py'], check=True)

            result_df = pd.read_csv('./predict/result/target_predictions.csv')
            result_df.columns = ['Absorption (nm)', 'Emission (nm)', 'Quantum Yield', 'Molar Abs. Coef.']
            result_html = result_df.to_html(classes='result-table', index=False, border=0)

            return render_template('prediction.html', result_table=result_html)

        except Exception as e:
            return render_template('prediction.html', error=str(e))

    return render_template('prediction.html')

# About 页面
@app.route('/about')
def about():
    return render_template('about.html')

# 模型运行逻辑
@app.route('/run_model', methods=['POST'])
def run_model():
    smiles = request.form['smiles']
    similarity_value = float(request.form['similarity_value'])

    input_file = os.path.join(app.config['UPLOAD_FOLDER'], 'target_m.csv')
    with open(input_file, 'w') as f:
        f.write("smiles\n")
        f.write(f"{smiles}\n")

    try:
        processing.process(similarity_value)

        # 提取前10个 predicted_label==1 的 smiles
        result_file = os.path.join(app.config['RESULT_FOLDER'], 'new_molecules.csv')
        smiles_list = []
        if os.path.exists(result_file):
            df = pd.read_csv(result_file)
            smiles_list = df[df['predicted_label'] == 1]['smiles'].head(20).tolist()

        return render_template('index.html', success=True, smiles_list=smiles_list)

    except RuntimeError as e:
        error_msg = str(e)
        if "无法拆分" in error_msg:
            return render_template('index.html', success=False, error="The molecular structure is too homogeneous to be effectively separated.")
        elif "没有找到匹配的规则" in error_msg:
            return render_template('index.html', success=False, error="No matching rules were found. Please lower the Similarity Value.")
        else:
            return render_template('index.html', success=False, error="Runtime error occurred.")

    except Exception as e:
        print(f"处理失败: {e}")
        return render_template('index.html', success=False, error="Model runtime failure: Input validation required.")

# 文件下载
@app.route('/download/<filename>')
def download_file(filename):
    return send_file(
        os.path.join(app.config['RESULT_FOLDER'], filename),
        as_attachment=True
    )

if __name__ == '__main__':
    app.run(debug=True)
