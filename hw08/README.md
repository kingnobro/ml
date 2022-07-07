## AutoEncoder

- 运行了 SampleCode，发现全连接的 AutoEncoder 准确率最高

- 修改 fcn 的模型架构。其中，中间表示的维度需要适当大一些，修改 ReLU 为 LeakyReLU，添加 BatchNorm1d

  ```python
  class fcn_autoencoder(nn.Module):
      def __init__(self):
          super(fcn_autoencoder, self).__init__()
          self.encoder = nn.Sequential(
              nn.Linear(64 * 64 * 3, 2048),
              nn.LeakyReLU(),
              nn.Linear(2048, 512),
              nn.LeakyReLU(),
              nn.Linear(512, 128),
          )
          
          self.decoder = nn.Sequential(
              nn.Linear(128, 512),
              nn.LeakyReLU(), 
              nn.Linear(512, 2048),
              nn.LeakyReLU(),
              nn.Linear(2048, 64 * 64 * 3), 
              nn.Tanh()
          )
  ```

- 学习率 warm up 和衰退

  ```python
  scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(train_dataloader), epochs=num_epochs)
  ```

- batch size 修改为 256

- 优化器修改为 AdamW



最后结果到达 strong baseline