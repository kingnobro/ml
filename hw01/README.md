### Training Loss

| Baseline | Score  |
| :------: | :----: |
|  Simple  | 1.6972 |
|  Medium  |        |
|  Strong  |        |
|   Boss   |        |



### 1. Feature Selection

|                 Feature                  |   Score    |
| :--------------------------------------: | :--------: |
|               All Features               |   1.6972   |
| id + state + features of the last 3 days |   1.3736   |
|   state + features of the last 3 days    |   1.0804   |
| **state + features of the last 2 days**  | **0.9552** |
|     state + features of the last day     |   4.6312   |
|       features of the last 2 days        |   0.9783   |



### 2. Different Model Architectures and Optimizers

|   Optimizer    | Score  |
| :------------: | :----: |
|      SGD       | 0.9552 |
| Adam (lr=1e-3) | 0.9581 |

```python
# Architecture, Score = 0.9462
self.layers = nn.Sequential(
    nn.Linear(input_dim, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)
```



### 3. L2 Regularization and Try More Parameters

| Weight Decay | Score  |
| :----------: | :----: |
|     1e-4     | 0.9445 |
|     1e-5     | 0.9449 |
|     1e-6     | 0.9437 |

other parameters
|     Parameters     | Value | Score  |
| :----------------: | :---: | :----: |
|        seed        |   7   | 0.7295 |
| data normalization | none  | 0.7169 |

