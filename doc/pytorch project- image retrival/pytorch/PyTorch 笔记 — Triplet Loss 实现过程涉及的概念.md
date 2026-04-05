

## Tensor 基础

PyTorch 的 `torch.Tensor` = 加强版 numpy array。

- 操作语法基本一致（切片、索引、四则运算）
- 多了两个关键能力：GPU 加速 + 自动求导（计算图）
- 不要随手 `.numpy()` 转换，会断掉梯度

```python
import torch
x = torch.tensor([1.0, 2.0, 3.0])  # 创建 tensor
x.shape    # 查看形状
x.dtype    # 查看数据类型
```

---

## unsqueeze — 插入维度

在指定位置插入一个新维度，本质上是为了触发广播。

```python
x = torch.tensor([1.0, 4.0, 9.0])  # shape: (3,)

x.unsqueeze(0)  # shape: (1, 3) — 行向量
x.unsqueeze(1)  # shape: (3, 1) — 列向量
```

---

## 广播机制 — Broadcasting

shape 不同的 tensor 做运算时，自动把小的那个"复制"扩展成大的形状。

```python
col = x.unsqueeze(1)  # (3, 1) — 每行复制 3 份 → (3, 3)
row = x.unsqueeze(0)  # (1, 3) — 每列复制 3 份 → (3, 3)

col + row
# 第 i 行第 j 列 = x[i] + x[j]
# 所有对一次算完，无需 for 循环
```

---

## 距离矩阵的计算

欧氏距离的展开公式：`||a - b||² = ||a||² + ||b||² - 2·(a·b)`

```python
# embeddings: (N, 128)

sq_norm = torch.sum(embeddings ** 2, dim=1)           # (N,)
dist_sq = sq_norm.unsqueeze(1) + sq_norm.unsqueeze(0) \
          - 2 * (embeddings @ embeddings.T)            # (N, N)

dist = torch.sqrt(torch.clamp(dist_sq, min=1e-12))    # (N, N)
```

---

## torch.clamp — 截断

把 tensor 的值限制在指定范围内。

```python
torch.clamp(x, min=0)          # 小于 0 的变成 0
torch.clamp(x, min=1e-12)      # 防止开根号时出现负数 → nan
torch.clamp(x, min=0, max=1)   # 同时限制上下界
```

开根号之前必须 clamp，因为浮点误差可能产生极小负数，`sqrt` 负数会得到 `nan`。

---

## Mask — 布尔矩阵

用广播比较生成同类/异类的 mask：

```python
# labels: (N,)
mask_pos = labels.unsqueeze(0) == labels.unsqueeze(1)  # (N, N) 同类为 True
mask_neg = ~mask_pos                                    # 异类为 True

mask_pos.fill_diagonal_(False)  # 排除自己跟自己
```

---

## .max / .min — 沿维度取极值

```python
values, indices = x.max(dim=1)   # 每行的最大值 + 索引
values, indices = x.min(dim=1)   # 每行的最小值 + 索引
```

注意：`torch.max(a, b)` 是两个 tensor 逐元素取较大值，跟 `.max(dim)` 是不同操作。

---

## Hard Positive / Hard Negative 的找法

```python
# Hard Positive：同类里距离最远的
anchor_positive_dist = dist * mask_pos.float()
hardest_positive, _ = anchor_positive_dist.max(dim=1)

# Hard Negative：异类里距离最近的（先把同类位置填成大数）
anchor_negative_dist = dist + (~mask_neg).float() * 1e9
hardest_negative, _ = anchor_negative_dist.min(dim=1)
```

---

## Triplet Loss

```python
loss = torch.clamp(hardest_positive - hardest_negative + margin, min=0.0)
loss = loss.mean()
```

含义：要求每个 anchor，它跟 positive 的距离比跟 negative 的距离**至少小 margin**。 违反约束才产生 loss，满足了就是 0。

---

## Type Hint

```python
from torch import Tensor

def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
    ...
```

只是提示，Python 不强制检查类型。但加了 IDE 自动补全更好用。