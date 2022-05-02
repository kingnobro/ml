- 修改 `concat_nframes = 23`（可以更大，但是笔记本内存不够用）
- 修改 `hidden_layers = 4`
- `hidden_dim = 1024`
- 增加训练的 epoch 到 15（次数太多会过拟合）
- AdamW 优化器新增 `weight_decay = 1e-3`
- `nn.Dropout(0.25)`，值太大会导致 training loss 降不下来，值太小会导致过拟合
- `learning_rate = 2e-3`
- `batch_size = 1024`
- 提高训练集比例至 0.9

## Model

```python
class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(),
            # BatchNorm 放在 Relu 后面
            nn.Dropout(0.25),
            nn.BatchNorm1d(output_dim),
        )

    def forward(self, x):
        x = self.block(x)
        return x
```

