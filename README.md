# LSPR 深度学习工程说明

本目录用于实现一个面向 LSPR 光谱分析的小样本深度学习方案。当前工程已经完成第一阶段的核心能力建设：基于理论仿真光谱与物理噪声增强，训练一个可迁移的 1D-ResNet Encoder，并通过 t-SNE 验证其在特征空间中对折射率等物理因素的分离能力。

这份 README 不只是脚本使用说明，也整合了 `Idea/` 目录中的总体计划和参考论文思路，用于指导后续从“理论预训练”走向“真实实验 few-shot 适配、物理约束校验以及工程化部署”。

## 1. 项目目标

项目的长期目标不是单纯得到一张分类图，而是构建一套“懂物理、能落地、可扩展”的 LSPR 智能分析框架。总体分为三层：

- 底层：基于 Mie 理论和物理噪声建模，学习稳定的光谱表征能力。
- 中层：利用少量真实实验数据，将仿真域特征迁移到真实检测任务。
- 上层：加入物理规则、可信约束和 GUI/软件集成，形成可用的分析系统。

## 2. 参考资料与方法映射

`Idea/` 文件夹中的资料对本工程的作用如下：

- `1.Chen_Meta-Baseline_Exploring_Simple_Meta-Learning_for_Few-Shot_Learning_ICCV_2021_paper.pdf`
  - 对应本项目的核心预训练思想。
  - 先做 whole-classification 预训练，得到通用 embedding，再用于 few-shot 任务适配。

- `4.Mind the (Data) Gap Evaluating Vision Systems in Small Data Applications.pdf`
  - 强化了“小数据场景下，强特征模型往往比复杂大模型更可靠”的工程判断。
  - 因此本项目优先采用 1D-ResNet，而不是直接上 Transformer 或多模态大模型。

- `5.SPARSE Data, Rich Results Few-Shot Semi-Supervised Learning via Class-Conditioned Image Translation.pdf`
  - 对应未来的稀疏数据增强方向。
  - 当极低浓度样本严重不足时，可考虑引入条件生成模型补足弱响应区域。

- `3.Lee_Temporal_Alignment-Free_Video_Matching_for_Few-shot_Action_Recognition_CVPR_2025_paper.pdf`
  - 对应未来的动态 Sensogram 分析方向。
  - 如果后续不只分析终点光谱，而要分析结合/解离过程，可借鉴无对齐时序匹配思想。

- `2.Architecture for a Trustworthy Quantum Chatbot（C4Q 2.0）.pdf`
  - 对应可信推理和规则校验层。
  - 深度学习预测结果最终应受到物理规律、浓度边界和峰位逻辑的约束。

- `LSPR深度学习.docx`
  - 是当前工程路线的总纲。
  - 里面明确了 MVP 最小闭环：先完成仿真预训练，再接入真实数据和物理逻辑约束。

## 3. 总体技术路线

按照目前计划，完整系统建议分成四大阶段：

### 阶段 A：理论光谱预训练

目标：训练一个“物理感知”的光谱 Encoder。

包含内容：

- 在 `(n, d)` 参数网格上批量生成理论消光光谱。
- 引入高斯噪声、基线漂移、峰位展宽等物理扰动。
- 使用 1D-ResNet + whole-classification 进行预训练。
- 保存 Encoder 权重，不保留分类头。
- 用 t-SNE 验证 embedding 是否能够区分不同折射率。

当前状态：已经完成并跑通。

### 阶段 B：真实实验 few-shot 适配

目标：让仿真域学到的特征迁移到真实实验域。

建议做法：

- 引入真实 BSA、患者样本或校准样本光谱。
- 构造 support/query 形式的小样本任务。
- 冻结或半冻结 Encoder，只训练轻量分类头或原型分类器。
- 在 128 维特征空间中使用余弦相似度完成分类或浓度映射。

阶段 B 是整个系统从“理论可行”走向“真实可用”的关键拐点。阶段 A 学到的是一个较强的光谱表示空间，但它主要来自仿真光谱和人工噪声；阶段 B 的任务，是让这个表示空间适配真实仪器、真实样品和真实实验流程带来的域偏移。

### 阶段 C：物理约束与可信判定

目标：避免黑盒模型输出违反物理规律的结果。

建议加入：

- 峰位红移方向检查。
- Beer-Lambert 或 Langmuir 区间边界约束。
- 高浓度饱和区停止线性外推。
- 传统峰值提取方法和 AI 输出进行交叉验证。

### 阶段 D：工程化与软件集成

目标：将模型封装为可实际使用的软件模块。

建议输出：

- 输入一条光谱，输出浓度/类别/置信度。
- 生成峰位图、对比图、推理日志。
- 接入现有 NanoSense/PyQt 界面与多线程工作流。

## 4. 当前已经实现的内容

当前代码已经完整覆盖阶段 A，也就是“基于 Meta-Baseline 的光谱特征预训练”。

### 4.1 参数网格与理论光谱生成

已实现：

- 按折射率 `n` 和粒径 `d` 组合生成基础类别。
- 调用 `PyMieScatt` 生成理论消光光谱。
- 读取金的光学常数 `data/au_johnson_nk.csv`。

相关文件：

- `lspr/spectra.py`
- `scripts/generate_pretrain_dataset.py`

### 4.2 物理噪声增强

已实现：

- 高斯白噪声
- 基线漂移
- 峰位/FWHM 展宽

相关文件：

- `lspr/noise.py`

### 4.3 1D-ResNet 预训练

已实现：

- 1D-ResNet Encoder
- 线性分类头
- CrossEntropyLoss + AdamW + CosineAnnealingLR
- GPU 训练
- 自动保存 Encoder 权重
- 自动记录 CSV / JSON / PNG 训练日志

相关文件：

- `lspr/model.py`
- `lspr/data.py`
- `scripts/train_pretrain.py`

### 4.4 t-SNE 流形验证

已实现：

- 加载预训练 Encoder
- 生成新的带噪声验证光谱
- 计算 embedding
- 使用 t-SNE 降维可视化

相关文件：

- `scripts/tsne_validate.py`

### 4.5 结果对比与实验管理

已实现：

- `small / mid / full` 三组模型对比
- embedding 指标计算
- 自动按时间戳归档实验输出
- 输出目录整理

相关文件：

- `scripts/compare_embedding_metrics.py`
- `scripts/eval_checkpoints.py`
- `scripts/run_pretrain_bundle.py`

## 5. 目录结构说明

```text
DeepLearning/Cnn/
├─ data/
│  ├─ au_johnson_nk.csv
│  ├─ pretrain_small/
│  ├─ pretrain_mid/
│  └─ pretrain_full/
├─ Idea/
│  ├─ 参考论文
│  └─ LSPR深度学习.docx
├─ lspr/
│  ├─ spectra.py
│  ├─ noise.py
│  ├─ model.py
│  └─ data.py
├─ outputs/
│  ├─ legacy/
│  └─ run_YYYYMMDD_HHMMSS/
├─ scripts/
│  ├─ generate_pretrain_dataset.py
│  ├─ train_pretrain.py
│  ├─ tsne_validate.py
│  ├─ plot_random_spectra.py
│  ├─ compare_embedding_metrics.py
│  ├─ eval_checkpoints.py
│  ├─ run_pretrain_bundle.py
│  ├─ prepare_real_dataset.py
│  ├─ prepare_paired_excel_dataset.py
│  ├─ train_fewshot.py
│  ├─ eval_fewshot.py
│  └─ predict_real_sample.py
└─ README.md
```

## 6. 环境依赖

建议使用 `py39` 环境，并安装：

- `numpy`
- `torch`
- `scikit-learn`
- `matplotlib`
- `PyMieScatt`

若要启用 GPU，请确保：

- 本机有 NVIDIA 显卡和驱动。
- PyTorch 为 CUDA 版本，而不是 CPU 版本。

## 7. 数据生成与训练流程

### 7.1 生成 full 数据集

```powershell
python -u scripts/generate_pretrain_dataset.py --out-dir data/pretrain_full --variants-per-class 1000
```

输出：

- `data/pretrain_full/pretrain_spectra.npy`
- `data/pretrain_full/pretrain_labels.npy`
- `data/pretrain_full/grid_meta.json`

### 7.2 单独训练 Encoder

```powershell
python -u scripts/train_pretrain.py --data-dir data/pretrain_full --epochs 10 --batch-size 256 --device cuda --num-workers 0 --log-interval 20
```

输出：

- `*.pth`
- `*_metrics.csv`
- `*_metrics.json`
- `*_metrics.png`

### 7.3 单独生成 t-SNE 图

```powershell
python -u scripts/tsne_validate.py --encoder outputs/lspr_encoder_v1.pth --out outputs/tsne_validation.png --samples-per-n 100 --unique-bases-per-n 12 --device cuda
```

输出：

- `tsne_validation.png`

## 8. 推荐工作流：一键归档运行

推荐直接使用下面这条命令，它会自动创建一个时间戳目录，并把本次实验所有关键文件放在同一个文件夹中。

```powershell
python -u scripts/run_pretrain_bundle.py --data-dir data/pretrain_full --device cuda
```

生成目录示例：

- `outputs/run_20260320_085725/`

目录内包含：

- `lspr_encoder_v1.pth`
- `train_metrics.csv`
- `train_metrics.json`
- `train_metrics.png`
- `tsne_validation.png`
- `run_manifest.json`

这样做的好处是：

- 每次实验互不覆盖
- 结果、图像、模型统一归档
- 后续论文写作和实验复现更方便

## 9. 当前实验现状

到目前为止，已经完成过：

- `small` 数据规模预训练与 t-SNE
- `mid` 数据规模预训练与 t-SNE
- `full` 数据规模 GPU 训练与归档

最近一次 full 训练的结果目录为：

- `outputs/run_20260320_085725/`

该次训练最终表现：

- `train_loss = 0.1433`
- `val_loss = 0.0621`
- `val_acc = 0.989`

这说明第一阶段的预训练已经取得非常强的分离能力，可以作为后续真实数据适配的基础模型。

## 10. 后续建议执行顺序

建议下一步按下面顺序推进：

1. 构建真实实验数据的 few-shot 支持集与查询集。
2. 基于当前 full Encoder 做真实域适配。
3. 为浓度回归或类别判断增加下游头部。
4. 加入物理约束与可信校验规则。
5. 最终嵌入现有软件系统。

## 11. 阶段 B 详细设计：真实实验 few-shot 适配

这一阶段建议作为下一轮开发重点。下面给出一个可以直接转成代码与实验 SOP 的实施方案。

### 11.1 阶段 B 的目标

阶段 B 的目标不是重新从零训练一个模型，而是在已经训练好的 full Encoder 基础上，用少量真实实验数据完成以下事情：

- 抵消仿真域到真实域之间的分布偏差。
- 保留阶段 A 学到的物理结构化表示能力。
- 让模型在极少标注样本下也能稳定完成分类或浓度推断。

### 11.2 推荐的真实数据组织方式

建议先将真实实验数据整理成统一表格，每一条样本至少包含以下字段：

- `sample_id`：样本编号
- `group`：类别或样本组，例如 `blank`、`bsa_low`、`bsa_mid`、`bsa_high`
- `concentration`：目标浓度，可为空
- `wavelength_start_nm`
- `wavelength_stop_nm`
- `spectrum`：等长采样后的光谱向量
- `source`：例如 `instrument_A`、`patient_batch_1`
- `split`：如 `support`、`query`、`test`

如果后续需要兼容当前 `.npy` 风格，也可以考虑以下目录格式：

```text
data/real_fewshot/
├─ metadata.csv
├─ spectra.npy
└─ wavelengths.npy
```

其中：

- `spectra.npy`：形状建议为 `[N, L]`
- `wavelengths.npy`：形状建议为 `[L]`
- `metadata.csv`：保存类别、浓度、批次、来源等信息

### 11.3 进入 few-shot 之前的真实数据预处理

真实数据在进入 Encoder 之前，建议统一经过以下处理：

- 波长轴重采样：统一到与预训练一致的长度和范围，例如 `400-800 nm / 400 points`
- 基线校正：去除仪器漂移和背景偏移
- 强度归一化：避免绝对光强差异影响 embedding
- 平滑或异常点过滤：只做轻量处理，避免破坏峰形
- 峰位检查：记录传统寻峰结果，用于后续可信约束

如果真实仪器的采样点不是 400 个，建议先插值到统一长度，再喂给 Encoder。

### 11.4 推荐的 few-shot 任务形式

可以优先做两类任务：

- 任务 1：少样本分类
  - 例如区分 `blank / low / medium / high`
  - 适合用原型网络式推理或余弦分类

- 任务 2：少样本回归
  - 例如浓度预测、折射率估计、粒径相关 proxy 预测
  - 适合在 embedding 上接一个轻量 MLP 或线性回归头

在项目推进顺序上，建议先做分类任务，跑通之后再做回归任务。因为分类更容易稳定，也更容易验证预训练是否有效。

### 11.5 建议的训练策略

阶段 B 推荐从简单到复杂分三层：

- 方案 A：冻结 Encoder，只做原型分类
  - 最稳，最适合作为第一版 few-shot 基线
  - 流程：support 求类原型，query 与各原型做余弦相似度

- 方案 B：冻结 Encoder，只训练轻量分类头
  - 适合少量标注样本略多时
  - 流程：128 维 embedding 接 `Linear` 或两层小 MLP

- 方案 C：半冻结 Encoder，微调最后一层或最后一个 block
  - 适合真实域偏移明显时
  - 风险是更容易过拟合，需要严格验证

推荐起步顺序：

1. 原型分类
2. 线性头
3. 半冻结微调

### 11.6 建议的 episode 构造方式

如果采用 Meta-Baseline 风格 few-shot 评估，建议定义如下 episode：

- `N-way`：3-way 或 4-way
- `K-shot`：1-shot、3-shot、5-shot
- 每个类别 query 数量：5 到 20

例如：

- `4-way 3-shot 10-query`
  - 4 个类别
  - 每类 support 3 条
  - 每类 query 10 条

这样可以系统性评估“标注样本数变化对性能的影响”。

### 11.7 建议实现的脚本

阶段 B 当前已实现以下脚本：

- `scripts/prepare_real_dataset.py`
  - 作用：将真实 CSV 整理成 few-shot 可用格式（`spectra.npy`、`labels.npy`、`metadata.csv`、`label_map.json`）
  - 示例：
    - `python -u scripts/prepare_real_dataset.py --input-csv data/your_real_data.csv --out-dir data/real_fewshot --label-col label`

- `scripts/prepare_paired_excel_dataset.py`
  - 作用：处理成对的 BSA/Ag Excel 光谱（第一列为波长，其余列名形如 `10.0ng/ml-BSA-01_01` 与 `10.0ng/ml-Ag-01_01`）
  - 默认输出 `delta = Ag - BSA` 表征，更适合抗原反应前后对比
  - 示例：
    - `python -u scripts/prepare_paired_excel_dataset.py --input-xlsx "C:/Users/Spc/Desktop/3.LSPR-code/LSPR_code/CEA_build/Hill_Filtered_Paired_Spectra_20pct.xlsx" --out-dir data/real_fewshot --representation delta --label-mode concentration`

- `scripts/train_fewshot.py`
  - 作用：加载预训练 Encoder，在真实数据上执行 few-shot 适配
  - 支持模式：
    - `prototype`：原型分类（不训练头部）
    - `linear_head`：训练线性分类头
  - 示例（原型模式）：
    - `python -u scripts/train_fewshot.py --data-dir data/real_fewshot --encoder-path outputs/run_YYYYMMDD_HHMMSS/lspr_encoder_v1.pth --mode prototype --k-shot 5 --n-query 20 --device cuda`
  - 示例（线性头）：
    - `python -u scripts/train_fewshot.py --data-dir data/real_fewshot --encoder-path outputs/run_YYYYMMDD_HHMMSS/lspr_encoder_v1.pth --mode linear_head --k-shot 5 --n-query 20 --epochs 80 --device cuda`
  - 注意：
    - 若某些类别样本数很少，`k_shot + n_query` 不能超过该类别样本数，否则会报错
    - 对类别不均衡数据可先用 `--k-shot 1 --n-query 1` 做保守验证

- `scripts/eval_fewshot.py`
  - 作用：批量评估不同 `K-shot` 与多 episode 的稳定性，输出 `episode_metrics.csv`、`summary_metrics.csv`、`summary_metrics.json` 和评估曲线图
  - 示例：
    - `python -u scripts/eval_fewshot.py --data-dir data/real_fewshot --encoder-path outputs/run_YYYYMMDD_HHMMSS/lspr_encoder_v1.pth --mode prototype --k-shots 1,3,5 --episodes 10 --device cuda`

- `scripts/predict_real_sample.py`
  - 作用：单条真实光谱推理（支持 prototype 和 linear_head 两种模式）
  - 示例（按索引推理，prototype）：
    - `python -u scripts/predict_real_sample.py --data-dir data/real_fewshot --encoder-path outputs/run_YYYYMMDD_HHMMSS/lspr_encoder_v1.pth --mode prototype --sample-index 0 --k-shot 5 --device cuda`
  - 示例（外部 .npy 光谱，linear head）：
    - `python -u scripts/predict_real_sample.py --data-dir data/real_fewshot --encoder-path outputs/run_YYYYMMDD_HHMMSS/lspr_encoder_v1.pth --mode linear_head --adapted-head outputs/fewshot_xxx/adapted_head.pth --fewshot-metrics-json outputs/fewshot_xxx/fewshot_metrics.json --spectrum-npy data/one_sample.npy --device cuda`

### 11.8 阶段 B 的输出文件建议

为了与阶段 A 保持统一，阶段 B 也建议输出到时间戳目录：

```text
outputs/fewshot_YYYYMMDD_HHMMSS/
├─ adapted_head.pth
├─ fewshot_metrics.csv
├─ fewshot_metrics.json
├─ confusion_matrix.png
├─ support_query_vis.png
└─ run_manifest.json
```

如果做回归任务，则建议补充：

- `prediction_vs_target.png`
- `regression_metrics.json`

### 11.9 阶段 B 的评估指标

分类任务建议记录：

- `accuracy`
- `macro_f1`
- `confusion_matrix`
- 不同 `K-shot` 下的均值和标准差

回归任务建议记录：

- `MAE`
- `RMSE`
- `R2`
- `Pearson/Spearman correlation`

同时建议补一个 embedding 可视化：

- few-shot 适配前 vs few-shot 适配后 的 t-SNE 对比

### 11.10 阶段 B 与阶段 C 的接口关系

阶段 B 输出的是“适配后的预测结果”，但这些结果还未必可信。阶段 C 会在阶段 B 输出的基础上再加一层物理约束，例如：

- 若折射率升高却出现峰位显著蓝移，则标记为异常
- 若预测浓度超过 Langmuir 饱和合理区间，则提示不可靠
- 若 AI 预测和传统峰值分析差异过大，则触发人工复核

换句话说：

- 阶段 B 负责“让模型看懂真实数据”
- 阶段 C 负责“让模型不要说违背物理的话”

## 12. 工程注意事项

- 若显存不足，可先降低 `--batch-size`，例如改为 `128`。
- 若只想快速调试，可降低 `--variants-per-class`。
- t-SNE 若较慢，可先用：

```powershell
python -u scripts/tsne_validate.py --encoder outputs/lspr_encoder_v1.pth --out outputs/tsne_quick.png --samples-per-n 50 --unique-bases-per-n 8 --device cuda
```

- 若使用 Windows，请优先在已激活环境中直接运行 `python -u ...`，避免 `conda run` 在部分场景下无输出或卡住。

## 13. README 的定位

这份文档的定位不是单纯的“怎么运行脚本”，而是：

- 项目总览
- 方法论说明
- 当前工程状态记录
- 后续研发路线图

后续如果进入阶段 B 和阶段 C，建议继续在本 README 中补充真实数据接口、few-shot 训练脚本、物理约束规则和软件集成方式。
