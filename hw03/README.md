### Transformation

对图像进行变换操作，使得每一个 epoch 中输入的图片都是不同的，避免过拟合。下面使用的方法是将许多个 transformation 拼接在一起，每一张图片都会经过所有函数的处理；也可以分别在多个数据上应用一种方法，然后将数据集 Concat 起来，但是这样数据量会呈线性增加

```python
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=1),
    transforms.RandomVerticalFlip(p=1),
    transforms.RandomGrayscale(0.5),
    transforms.RandomSolarize(threshold=192.0),
    transforms.ColorJitter(brightness=.5,hue=0.5),
    transforms.RandomRotation(degrees=(0, 180)),
    transforms.RandomInvert(),
    transforms.ToTensor(),
])
```



### Models

选用了比较深的模型 [densenet201](https://blog.csdn.net/qq_33287871/article/details/108964027?spm=1001.2101.3001.6650.4&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-4-108964027-blog-123908742.pc_relevant_antiscanv3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-4-108964027-blog-123908742.pc_relevant_antiscanv3&utm_relevant_index=9)



### Mixup

将两张图片混合在一起。需要修改 Dataset 中 `__getitem__()` 方法，使其返回两个图片线性混合的结果，并且 label 需要变成一个向量；`CrossEntropyLoss` 也需要手动修改

```python
def __getitem__(self,idx):
    # 只有训练时需要 mixup
    if self.train:
        fname1 = self.files[idx]
        fname2 = self.files[np.random.randint(0, len(self.files))]

        im1 = Image.open(fname1).resize((128, 128))
        im2 = Image.open(fname2).resize((128, 128))
        im = Image.blend(im1, im2, self.lam)
        im = self.transform(im)

        label1 = int(fname1.split("/")[-1].split("_")[0])
        label2 = int(fname2.split("/")[-1].split("_")[0])
        label = [label1, label2]
    else:
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1 # test has no label
    return im,label

# loss 要计算两个标签
loss = (1.0 - lam) * criterion(logits, labels[0].to(device)) + lam * criterion(logits, labels[1].to(device))

```



### Test Time Augmentation (Ensemble)

在样例代码中，对测试数据的 transformation 都是相同的（只有 Resize）。可以将运用在训练数据上的 transform 方法也运用在测试数据上，生成多个预测结果，然后将结果加权平均



### Residual Connection Implementation



```python
from torch import nn
class Residual_Network(nn.Module):
    def __init__(self):
        super(Residual_Network, self).__init__()
        
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
        )

        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
        )

        self.cnn_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
        )

        self.cnn_layer4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
        )
        self.cnn_layer5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
        )
        self.cnn_layer6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(256* 32* 32, 256),
            nn.ReLU(),
            nn.Linear(256, 11)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 11]

        # Extract features by convolutional layers.
        x1 = self.cnn_layer1(x)
        x1 = self.relu(x1)
        residual = x1
        
        x2 = self.cnn_layer2(x1)
        x2 += residual
        x2 = self.relu(x2)
        
        x3 = self.cnn_layer3(x2)
        x3 = self.relu(x3)
        residual = x3
        
        x4 = self.cnn_layer4(x3)
        x4 += residual
        x4 = self.relu(x4)
        
        x5 = self.cnn_layer5(x4)
        x5 = self.relu(x5)
        residual = x5
        
        x6 = self.cnn_layer6(x5)
        x6 += residual
        x6 = self.relu(x6)
        
        # The extracted feature map must be flatten before going to fully-connected layers.
        xout = x6.flatten(1)

        # The features are transformed by fully-connected layers to obtain the final logits.
        xout = self.fc_layer(xout)
        return xout
```

