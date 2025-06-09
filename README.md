# Fluor-tools

**Fluor-tools** is an integrated web-based and local computational platform for dye structure optimization and property prediction. It includes two core modules:

* **NIRFluor-opt**: Converts non-NIR dyes to NIR dyes using a rule-based strategy.
* **Fluor-RLAT**: Predicts photophysical properties of dyes using a multimodal deep learning model.

---

## 🔗 Online Usage

You can directly use Fluor-tools at the official website:
👉 [https://lmmd.ecust.edu.cn/Fluor-tools/](https://lmmd.ecust.edu.cn/Fluor-tools/)

> **Note:** If you do not have a programming background or only intend to use the models, we strongly recommend using the web interface. If you wish to perform batch processing or customize computations, please follow the local setup instructions below.

---

## ⚙️ Environment Setup

Create a Python environment and install dependencies:

```bash
conda create -n dye37 python=3.7
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install pandas==1.3.0
conda install -c anaconda scikit-learn
conda install -c conda-forge lightgbm
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple xgboost
pip install dgllife==0.2.8
pip install rdkit-pypi
pip install dgl==1.1.2+cu117 -f https://data.dgl.ai/wheels/cu117/repo.html
```

### 🔧 DGL Compatibility Fix

If you encounter errors in DGL's model zoo (especially in `attentivefp.py`):

* Replace `fn.copy_edge` with `fn.copy_e`
* Replace `fn.src_mul_edge` with `fn.u_mul_e`

---

## 🚀 How to Use NIRFluor-opt

```bash
cd NIRFluor-opt and web
```

1. Customize your target molecule and `Similarity_Value` hyperparameter in `run.py`
2. Run the script:

```bash
python run.py
```

3. Optimized NIR dye molecules will be saved to:

```bash
./results/merged_file_pred.csv
```

> ⚠️ Extracting transformation rules from scratch using MMP may take up to one week.


---

## 🔮 How to Use Fluor-RLAT

```bash
cd Fluor-RLAT
```

1. Customize `run.py` with your molecules and solvent information
2. Run the script:

```bash
python run.py
```

3. Prediction results will be saved to:

```bash
./result/target_predictions.csv
```

> 🔁 For batch predictions, simply modify the loop logic in `run.py`

## 🧠 Fluor-tools Architecture

![image](https://github.com/wenxiang-Song/fluor_tools/blob/main/Figure1.png?raw=true)
---

## 📬 Contact

For issues or inquiries, please open an [Issue] or contact the developers via the project homepage.

---

Enjoy using **Fluor-tools** for intelligent dye discovery! 🌈
