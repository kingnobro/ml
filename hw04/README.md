- 在本机跑代码的时候，运行到 `train_iterator = iter(train_loader)` 这行代码时会卡住不动。将 `n_workers` 修改为 0
- 要将 `cuda` 修改为 `cuda:0` 才能用上 GPU



- `n_head = 4`
- encoder layer 层数设置为 6
