### 1. 特征选择

使用 Pearson 相关系数选择出系数值大于 0.8 的特征（大约为 24 个）

```python
train_data = pd.read_csv('./covid.train.csv')
print(train_data.corr()['tested_positive'].sort_values(ascending=False).index)
```



### 2. 修改网络架构

将 `nn.ReLU()` 替换为 `nn.LeakyReLU()`；降低神经元的个数



### 3. 修改优化器

换成 Adam，`weight_decay=1e-3`



### 4. 增大训练集的占比

训练集占比为 0.9
