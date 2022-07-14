## Network Compression

- Complete KL Divergence loss function for knowledge distillation
- Modify model architecture with depth-wise and point-wise convolutions

```python
self.cnn = nn.Sequential(
    dwpw_conv(3, 32, 3),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    dwpw_conv(32, 32, 3),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(2, 2, 0),

    dwpw_conv(32, 64, 3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    dwpw_conv(64, 64, 3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2, 2, 0),

    dwpw_conv(64, 256, 3),
    nn.BatchNorm2d(256),
    nn.ReLU(),
    dwpw_conv(256, 128, 3),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.MaxPool2d(2, 2, 0),

    dwpw_conv(128, 128, 3),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.MaxPool2d(2, 2, 0),

    nn.AdaptiveAvgPool2d((1, 1)),
)
```

```text
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1          [-1, 3, 222, 222]              30
            Conv2d-2         [-1, 32, 222, 222]             128
       BatchNorm2d-3         [-1, 32, 222, 222]              64
              ReLU-4         [-1, 32, 222, 222]               0
            Conv2d-5         [-1, 32, 220, 220]             320
            Conv2d-6         [-1, 32, 220, 220]           1,056
       BatchNorm2d-7         [-1, 32, 220, 220]              64
              ReLU-8         [-1, 32, 220, 220]               0
         MaxPool2d-9         [-1, 32, 110, 110]               0
           Conv2d-10         [-1, 32, 108, 108]             320
           Conv2d-11         [-1, 64, 108, 108]           2,112
      BatchNorm2d-12         [-1, 64, 108, 108]             128
             ReLU-13         [-1, 64, 108, 108]               0
           Conv2d-14         [-1, 64, 106, 106]             640
           Conv2d-15         [-1, 64, 106, 106]           4,160
      BatchNorm2d-16         [-1, 64, 106, 106]             128
             ReLU-17         [-1, 64, 106, 106]               0
        MaxPool2d-18           [-1, 64, 53, 53]               0
           Conv2d-19           [-1, 64, 51, 51]             640
           Conv2d-20          [-1, 256, 51, 51]          16,640
      BatchNorm2d-21          [-1, 256, 51, 51]             512
             ReLU-22          [-1, 256, 51, 51]               0
...
Forward/backward pass size (MB): 161.50
Params size (MB): 0.31
Estimated Total Size (MB): 162.39
----------------------------------------------------------------
```

