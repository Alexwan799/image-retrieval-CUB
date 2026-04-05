
到现在用到或者提到过的 PyTorch 操作，按照你写 `model.py` 的顺序来：

**导包**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
```

**定义模型骨架**

```python
class EmbeddingModel(nn.Module):   # 所有模型都继承这个
    def __init__(self):
        super().__init__()         # 固定写法，初始化父类
```

**加载预训练模型**

```python
models.resnet50(weights='IMAGENET1K_V1')  # 加载带 ImageNet 权重的 ResNet-50
backbone.fc = nn.Identity()               # 把 fc 层换成透传，相当于砍头
```

**定义层**

```python
nn.Linear(2048, 128)      # 全连接层（dense layer）
nn.BatchNorm1d(128)       # 批归一化
nn.ReLU()                 # 激活函数
nn.Sequential(...)        # 把多层打包成流水线
```

**forward 里的操作**

```python
x = self.backbone(x)              # 数据过 backbone
x = self.embedding_head(x)        # 数据过 embedding head
x = F.normalize(x, p=2, dim=1)   # L2 归一化，每个向量长度变成 1
```

**验证用**

```python
torch.backends.mps.is_available()   # 检查 M1 MPS 是否可用
```

就这些。不多，但够你写完 `model.py` 了。去写吧。