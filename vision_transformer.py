import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datetime import datetime


device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")


class patchembedding(nn.Module):
    def __init__(self, patch_size, n_channels, d_model):
        super(patchembedding, self).__init__()

        self.patch_size = patch_size
        self.n_channels = n_channels
        self.d_model = d_model

        self.conv2d = nn.Conv2d(self.n_channels, self.d_model, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        # print(x.size())
        x = self.conv2d(x)  # size of x is (B,d_model,H,W)
        x = x.flatten(2)  # size of x is (B,d_model,H*W)
        x = x.transpose(-2, -1)  # size of x is (B,H*W,d_model)
        # print(x.size())
        return x


class positionalencoding(nn.Module):
    def __init__(self, max_sequence, d_model):
        super(positionalencoding, self).__init__()

        self.max_sequence = max_sequence
        self.d_model = d_model

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        pos_mat = torch.zeros(max_sequence + 1, d_model)

        for pos in range(self.max_sequence + 1):
            for i in range(self.d_model):
                if i % 2 == 0:
                    pos_mat[pos][i] = np.sin(pos / (10000 ** (i / self.d_model)))
                else:
                    pos_mat[pos][i] = np.cos(pos / (10000 ** ((i - 1) / self.d_model)))
        self.register_buffer('pos_mat', pos_mat.unsqueeze(0))

    def forward(self, x):
        B = x.shape[0]

        tokens_batch = self.cls_token.expand(B, -1, -1)
        x = torch.cat((tokens_batch, x), dim=1)

        # print(tokens_batch.size(),self.pos.size())
        # print(self.pos_mat.size())
        # print(x.size())
        x = x + self.pos_mat

        return x


class AttentionHead(nn.Module):
    def __init__(self, d_model):
        super(AttentionHead, self).__init__()

        self.d_model = d_model

        self.scaled = self.d_model ** 0.5

        self.qkv = nn.Linear(self.d_model, self.d_model * 3)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.d_model).permute(2, 0, 1, 3)
        Q, K, V = qkv.unbind(0)

        score = F.softmax(Q @ K.transpose(-2, -1) / self.scaled, dim=-1)
        return score @ V


class MutiheadsAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MutiheadsAttention, self).__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = self.d_model // self.num_heads

        self.attention = nn.ModuleList([AttentionHead(d_model=self.d_model) for _ in range(self.num_heads)])

        self.linear = nn.Linear(self.d_model * self.num_heads, self.d_model)

    def forward(self, x):
        attention = torch.cat([head(x) for head in self.attention], dim=-1)

        out = self.linear(attention)

        return out


class FeedForward(nn.Module):
    def __init__(self,d_model,d_ratio):
        super(FeedForward,self).__init__()

        self.linear1 = nn.Linear(d_model,d_model * d_ratio)
        self.linear2 = nn.Linear(d_model * d_ratio,d_model)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self,x):
        x=self.dropout2(self.linear2(self.dropout1(self.gelu(self.linear1(x)))))
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ratio, num_heads):
        super(EncoderLayer, self).__init__()

        self.d_model = d_model
        self.d_ratio = d_ratio
        self.num_heads = num_heads

        self.mutiattention = MutiheadsAttention(d_model=self.d_model, num_heads=self.num_heads)
        self.ffw = FeedForward(d_model=self.d_model, d_ratio=self.d_ratio)

        self.norm1 = nn.LayerNorm(normalized_shape=[self.d_model])
        self.norm2 = nn.LayerNorm(normalized_shape=[self.d_model])

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x):
        out1 = x + self.dropout1(self.mutiattention(self.norm1(x)))

        out2 = out1 + self.dropout2(self.ffw(self.norm2(out1)))

        return out2


class VisionTransformer(nn.Module):
    def __init__(self, image_size, d_model, d_ratio, num_heads, patch_size, max_sequence, n_channels, n_layers,
                 classes,device):
        super(VisionTransformer, self).__init__()

        assert image_size[0] % patch_size == 0 and image_size[1] % patch_size == 0
        assert d_model % num_heads == 0

        self.image_size = image_size
        self.d_model = d_model
        self.d_ratio = d_ratio
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.n_layers = n_layers
        self.classes = classes
        self.max_sequence = max_sequence
        self.grid_size = image_size[0] // patch_size
        self.device= device

        self.patchembedding = patchembedding(patch_size=self.patch_size, n_channels=self.n_channels,
                                             d_model=self.d_model)
        self.positionalencoding = positionalencoding(max_sequence=self.max_sequence, d_model=self.d_model)
        self.encoder = nn.Sequential(
            *[EncoderLayer(d_model=self.d_model, d_ratio=self.d_ratio, num_heads=self.num_heads)
              for _ in range(self.n_layers)])

        self.classfier = nn.Sequential(*[nn.Linear(self.d_model, self.classes)]
                                       )

    def forward(self, x):
        x = self.patchembedding(x)
        x = self.positionalencoding(x)
        x = self.encoder(x)
        x = self.classfier(x[:, 0])
        return x




# image_size = (32, 32)  # 调整图像尺寸
# n_channels = 1  # 输入图像的channnel
# d_model = 16  # 每个patch转换为多少维度的向量
# d_ratio = 4  # encoder中线性层过渡维度
# num_heads = 4  # mutiattention的head数量
# patch_size = 16
# n_layers = 5  # encoder层数
# classes = 10  # 要识别的图片数量
# batch_size = 64  # 总共有59904(64*936)张图片
# max_sequence = 4  # max_sequence = (image_size[0]//patch_size)**2
#
# transform = transforms.Compose([
#     transforms.Resize(image_size),  # 调整尺寸
#     transforms.ToTensor(),  # 转换为张量
#     transforms.Normalize((0.1307,), (0.3081,))  # 归一化到[0,1]
# ])
#
# train_dataset = torchvision.datasets.MNIST(
#     root='../datasets',
#     train=True,
#     download=True,
#     transform=transform)
#
# train_loader = DataLoader(train_dataset,
#                         shuffle=True,
#                         batch_size=batch_size,
#                         pin_memory=True)
#
# test_dataset = torchvision.datasets.MNIST(
#     root='../datasets',
#     train=False,
#     download=True,
#     transform=transform)
#
# test_loader = DataLoader(test_dataset,
#                         shuffle=False,
#                         batch_size=batch_size)
#
# model = VisionTransformer(image_size=image_size,
#                 d_model=d_model,
#                 d_ratio=d_ratio,
#                 num_heads=num_heads,
#                 patch_size=patch_size,
#                 n_channels=n_channels,
#                 n_layers=n_layers,
#                 classes=classes,
#                 max_sequence=max_sequence,
#                 device=device).to(device)
#
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.005)
#
# def train(epoch):
#     running_loss = 0.0
#     for batch_idx, data in enumerate(train_loader, 0):
#
#         inputs, target = data
#         inputs, target = inputs.to(device), target.to(device)
#         optimizer.zero_grad()
#
#         outputs = model(inputs)
#         # print(outputs.size(),target.size())
#         loss = criterion(outputs, target)
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#
#         if batch_idx % 300 == 0:
#             print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
#             running_loss = 0.0
#
# def test():
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in test_loader:
#             images, labels = data
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, dim=1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#         print('Accuracy on test set: %.1f %%' % (100 * correct / total))
#
# if __name__ == '__main__':
#     for epoch in range(30):
#         start_time = datetime.now()
#         train(epoch)
#         test()
#         end_time = datetime.now()
#         time_diff = end_time - start_time
#         print("time span:", time_diff)

