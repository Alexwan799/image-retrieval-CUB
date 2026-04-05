# PyTorch Tensor 维度笔记

---

## 1. Dimension vs Size

| 概念 | 说明 |
|------|------|
| `dim` | 轴的**编号**（索引），从0开始 |
| `size` | 那个轴有**多少个元素** |

```python
t = torch.randn(4, 5)  # shape: [4, 5]

# dim=0, size=4   → 行方向
# dim=1, size=5   → 列方向
```

> ⚠️ **容易混淆点：数学里的"维"和PyTorch的`dim`不是一回事**
>
> 数学里说"128维向量"，意思是这个向量有128个分量，对应PyTorch里`shape=[128]`，
> 这个128是size，不是dim。
>
> PyTorch的`dim`是轴的**编号**（0、1、2...），说的是"沿哪个轴操作"，不是向量有多长。
>
> 简单记法：**数学的"维" = PyTorch的size；PyTorch的dim = 轴的索引编号**

---

## 2. torch.sum(tensor, dim=N) 之后shape变成什么

> ⚠️ **容易踩坑：`torch.sum(..., dim=1)` 之后是 `[B]`，不是 `[B, 1]`**

被操作的那个dim会被**完全消掉**，不是变成size=1，是直接没了。

```python
t = torch.randn(B, 128)       # shape: [B, 128]

torch.sum(t, dim=1)            # → [B]      ← dim=1消失了
torch.sum(t, dim=1, keepdim=True)  # → [B, 1]  ← keepdim=True才保留轴
```

为什么默认消掉？因为128个数加成了1个数，那个轴存在的意义没了。

如果你想保留shape方便后续广播，加`keepdim=True`。

实际例子——计算每个embedding的L2 norm平方：

```python
sq_norm = torch.sum(embeddings ** 2, dim=1)  # [B, 128] → [B]
# 等价于：‖x‖² = x₁² + x₂² + ... + x₁₂₈²
```

---

## 3. `[B]` 和 `[B, 1]` 数学上一样，PyTorch里不一样

> ⚠️ **容易混淆点：数学上等价，但PyTorch广播行为完全不同**

数学上，一个长度为B的向量不管怎么包装语义是一样的。
但PyTorch的shape是严格的，`[B]`和`[B, 1]`触发不同的广播：

```python
a = torch.tensor([1., 2., 3.])          # shape [3]
b = torch.tensor([[1.], [2.], [3.]])    # shape [3, 1]

a + a   # → [3]      element-wise，没有广播
b + b   # → [3, 1]   element-wise，没有广播

a + b   # → [3, 3]   ← 广播！因为形状不对称
```

这就是为什么triplet loss里要手动`unsqueeze`——**故意制造形状不对称来触发广播**。

---

## 4. tensor ** 2 是什么

Element-wise操作，每个元素独立平方，**形状不变**。

```python
t = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])   # shape: [2, 2]

t ** 2
# → tensor([[ 1.,  4.],
#           [ 9., 16.]])         # shape 不变，还是 [2, 2]
```

PyTorch里凡是两个同shape的tensor直接做 `+`、`-`、`*`、`**`，都是element-wise，逐元素操作，形状不变。

---

## 5. Broadcasting（广播）规则

> ⚠️ **容易卡住点：为什么 `[3] + [3,1]` 能得到 `[3,3]`？**

### 规则

**第一步：从右边对齐两个shape。**

**第二步：逐个dim比较，满足以下任一条件才能广播：**
- 两边size相同 → 不变
- 其中一边size=1 → 扩展成另一边的size
- 其中一边这个dim不存在 → 自动补1再扩展

**第三步：每个dim取较大的size，得到输出shape。**

### 例子一：`[3] + [3, 1]` → `[3, 3]`

```
     [3]   →  右边对齐，左边自动补1  →  [1, 3]
   [3, 1]                               [3, 1]

dim=0:  1 vs 3  → size=1，扩展成3
dim=1:  3 vs 1  → size=1，扩展成3
输出: [3, 3]
```

展开后实际发生的事：

```
[1, 3]  →  复制3行  →  [[1, 2, 3],
                        [1, 2, 3],
                        [1, 2, 3]]   shape: [3, 3]

[3, 1]  →  复制3列  →  [[1, 1, 1],
                        [2, 2, 2],
                        [3, 3, 3]]   shape: [3, 3]
```

两个`[3, 3]`相加，第i行第j列 = `a[j] + b[i]`，是所有组合的pairwise结果。

### 例子二：`[B] + [B]` → `[B]`（不会广播）

```
[B]
[B]
→ 两边完全一样，直接element-wise → [B]
```

不会扩展，结果还是`[B]`，只是逐元素相加，不是pairwise。

### 例子三：shape不兼容会报错

```
[3, 4] + [3, 5]
→ dim=1: 4 vs 5，两边都不是1 → RuntimeError
```

---

### unsqueeze：手动控制广播方向

`unsqueeze(n)` 在第n个位置插入一个size=1的轴，**目的是触发特定方向的广播**。

```python
a = torch.tensor([1., 2., 3.])  # shape: [3]

a.unsqueeze(0)  # → [1, 3]   第0个位置插入1
a.unsqueeze(1)  # → [3, 1]   第1个位置插入1
```

记法：**想让哪个维度在行，就`unsqueeze(1)`；想让哪个在列，就`unsqueeze(0)`。**

```
unsqueeze(1) → [N, 1] → N在行方向
unsqueeze(0) → [1, N] → N在列方向
```

---

### 实战一：Triplet Loss 的 Pairwise 距离矩阵（B×B）

目标：B个sample两两之间的距离，结果`[B, B]`。

公式：`‖a - b‖² = ‖a‖² + ‖b‖² - 2aᵀb`

```python
sq_norm = torch.sum(embeddings ** 2, dim=1)  # [B, 128] → [B]

dist_sq = sq_norm.unsqueeze(1)               # [B] → [B, 1]  B在行
        + sq_norm.unsqueeze(0)               # [B] → [1, B]  B在列
        - (embeddings @ embeddings.T) * 2    #        [B, B]

# [B, 1] + [1, B] → [B, B]
# 第i行第j列 = ‖sample_i - sample_j‖²
```

---

### 实战二：ProxyNCA 的 Sample-to-Proxy 距离矩阵（B×C）

目标：B个sample 到 C个proxy 的距离，结果`[B, C]`。

```python
norm_a = torch.sum(embed_a ** 2, dim=1)  # [B]
norm_b = torch.sum(embed_b ** 2, dim=1)  # [C]

dist = norm_a.unsqueeze(1)               # [B] → [B, 1]  B在行
     + norm_b.unsqueeze(0)               # [C] → [1, C]  C在列
     - 2 * embed_a @ embed_b.T           #        [B, C]

# [B, 1] + [1, C] → [B, C]
# 第i行第j列 = sample_i 到 proxy_j 的距离
```

> ⚠️ **方向写反是常见bug：**
>
> `norm_a.unsqueeze(0) + norm_b.unsqueeze(1)` → `[1,B] + [C,1]` → `[C, B]`
>
> 而 `embed_a @ embed_b.T` 是 `[B, C]`，B≠C时直接报错。
> 出错时先检查unsqueeze方向，确认结果shape是否符合预期。

---

## 6. 三维Tensor示例（以后会遇到）

```python
# 一个batch的图片
t = torch.randn(32, 3, 224)

# dim=0, size=32   → batch size（多少张图）
# dim=1, size=3    → channel（RGB）
# dim=2, size=224  → 宽度
```
