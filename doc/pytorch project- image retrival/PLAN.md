# pytorch-project - 执行蓝图

> 状态：DRAFT — 等 Alex 确认后正式启动

## [Goal] 最终交付物
一个小而完整的 **Image Retrieval** PyTorch 项目，可放在 GitHub 展示。
- 输入一张 query 图，从图库中检索出最相似的 K 张图
- 使用 metric learning 训练 embedding 模型
- 有清晰的评估指标和可视化展示

## [Architecture] 项目架构

> 完整流程图见 `architecture.mermaid`

```
整体数据流：

  图片 ──→ ResNet-50 (backbone) ──→ 128-d Embedding ──→ Triplet Loss 训练
                                          │
                                          ▼
  Query 图 ──→ 同一个模型 ──→ query embedding ──→ cosine similarity ──→ Top-K 检索结果
                                                        ▲
                                          gallery embeddings（test set 全部图的向量）
```

### Repo 目录结构
```
image-retrieval/
├── configs/
│   └── default.yaml          # 超参数配置（lr, margin, embed_dim, epochs...）
├── data/
│   └── cub200/               # 数据集（git ignore）
├── src/
│   ├── dataset.py            # CUBDataset 类 + transforms
│   ├── model.py              # EmbeddingModel（backbone + head）
│   ├── losses.py             # TripletLoss, 后续加其他 loss
│   ├── sampler.py            # Triplet 采样策略（batch-hard mining）
│   ├── train.py              # 训练循环
│   ├── evaluate.py           # 提取 embedding + Recall@K + mAP
│   └── visualize.py          # 检索结果可视化
├── notebooks/
│   └── exploration.ipynb     # 数据探索 + 实验记录
├── checkpoints/              # 模型权重（git ignore）
├── results/                  # 实验结果图表
├── requirements.txt
└── README.md
```

每个文件对应一个清晰的职责，**你改一个地方不会牵连其他地方**。这个结构从 P0 开始就建好，后面只往里填内容。

## [Phases] 里程碑拆解

### P0: 环境搭建 + 数据准备（1-2天）
- [x] 建 repo，初始化项目结构（`data/`, `src/`, `notebooks/`, `configs/`）
- [x] 选定数据集并下载（推荐起步：**CUB-200-2011** 鸟类数据集，200类，~12k张图，体积小，benchmark经典）
- [x] 写 `Dataset` 类，能正确加载图片+标签
- [x] 写一个 dataloader 测试脚本，print 出 batch shape + 可视化几张样本
- **完成标志**：`python test_dataloader.py` 能跑通，能看到图片

### P1: Baseline 模型 + 训练循环（3-5天）
- [ ] 选一个 pretrained backbone（ResNet-50），砍掉分类头，接一个 embedding layer（128-d）
- [ ] 实现 Triplet Loss（anchor, positive, negative）
- [ ] 实现 triplet 采样策略（先用 batch-hard mining）
- [ ] 写训练循环：train loop + validation + checkpoint 保存
- [ ] 跑一轮训练，确认 loss 在下降
- **完成标志**：训练不崩，loss 收敛，能保存模型

### P2: 评估 + 检索 pipeline（2-3天）
- [ ] 对 test set 所有图提取 embedding
- [ ] 实现 KNN 检索（先用 brute-force cosine similarity，不急着上 FAISS）
- [ ] 实现评估指标：Recall@1, Recall@5, Recall@10, mAP
- [ ] 写可视化脚本：给一张 query 图，展示 top-5 检索结果（对/错用绿/红框标注）
- **完成标志**：能输出 Recall@K 数字 + 好看的检索结果可视化

### P3: 改进 + 实验对比（3-5天）
- [ ] 换 loss 函数对比（Triplet → Contrastive 或 ArcFace 或 ProxyNCA）
- [ ] 调 embedding 维度（64 vs 128 vs 256）
- [ ] 记录实验结果到表格（loss 类型 × embedding dim × Recall@K）
- [ ] 分析：哪些 query 检索得好，哪些差？failure case 分析
- **完成标志**：有一张实验对比表，有 failure case 分析

### P4: 文档 + GitHub 展示打磨（1-2天）
- [ ] 写 README：问题定义、方法、结果、可视化
- [ ] 整理代码结构，加 docstring
- [ ] 确保 `pip install -r requirements.txt` + `python train.py` 能一键跑通
- **完成标志**：一个陌生人 clone 你的 repo 能看懂你做了什么

## [Constraints/Rules]
- 第一版只做 **image-to-image retrieval**，不做 text-to-image
- 不上 CLIP、不上 FAISS、不搭前端、不搞 Docker——这些是"以后可以加"的东西，不是 MVP
- 不追求 SOTA，追求 **完整闭环 + 清晰展示**
- 每个 Phase 结束时跟我 check-in，不要闷头跑

## [Timeline] 预估总时长
- 乐观：10天
- 现实：2-3周
- 这不是 deadline，是方向感。做慢了不丢人，不动才丢人。
