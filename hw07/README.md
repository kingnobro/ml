## BERT - QA

- 使用 hfl/chinese-macbert-base，超过 1G 的模型显存不够用
- 学习率线性下降
- gradient accumulation，修改 batch size 为 8 × 8
- 修改 doc_stride
- 修改段落的起始位置和终止位置，避免网络以为答案永远在段落中间
- 最后取消 validation 步骤，将数据和训练数据合并在一起
- 将 num_epoch 修改为 2，会出现过拟合但是 kaggle 上的分数会略微提高
- postprocessing：假如 start index > end index，就直接 continue
- 答案中还会出现 [UNK]，表示是生僻字。不知道怎么解决