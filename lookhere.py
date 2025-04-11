import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import squareform, pdist, cdist
from vision_transformer import MutiheadsAttention, EncoderLayer, VisionTransformer,AttentionHead
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datetime import datetime

device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")


#定义模型
class EncoderLayerWithBiases(EncoderLayer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.mutiattention = MutiheadsAttentionwithbias(d_model=self.d_model, num_heads=self.num_heads)



class AttentionWithBiases(AttentionHead):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_bias_map(self, bias_map):
        self.bias_map = bias_map


    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.d_model).permute(2, 0, 1, 3)
        Q, K, V = qkv.unbind(0)

        score = F.softmax(Q @ K.transpose(-2, -1) / self.scaled +self.bias_map ,dim=-1)
        return score @ V


class MutiheadsAttentionwithbias(MutiheadsAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.attention = nn.ModuleList(
            [AttentionWithBiases(d_model=self.d_model) for _ in range(self.num_heads)])


class LookHere(VisionTransformer):
    def __init__(self,**kwargs):

        super().__init__(**kwargs)
        self.encoder =nn.Sequential(
            *[EncoderLayerWithBiases(d_model=self.d_model, d_ratio=self.d_ratio, num_heads=self.num_heads)
              for _ in range(self.n_layers)])
        self.set_pos_embed(self.grid_size)
        self.to(self.device)

    def set_pos_embed(self, grid_size, global_slope=None):
        grid_size = (grid_size, grid_size)


        num_patch = int(grid_size[0] * grid_size[1])
        bias_maps = create_lh_bias_tensor(
            Interval=(1.5,0.5),grid_size=grid_size, n_layers=self.n_layers, num_heads=self.num_heads
        ).to(
            self.device
        )  # (n_layers, num_heads, grid_size^2, grid_size^2)
        zeros_1 = torch.zeros(
            size=(self.n_layers, self.num_heads, num_patch, 1),
            dtype=torch.float,
            device=self.device,
        )  # (n_layers, num_heads, grid_size^2, 1)
        bias_maps = torch.cat(
            [zeros_1, bias_maps], dim=-1
        )  # (n_layers, num_heads, grid_size^2, grid_size^2+1)
        zeros_2 = torch.zeros(
            size=(self.n_layers, self.num_heads, 1, (num_patch + 1)),
            dtype=torch.float,
            device=self.device,
        )  # (n_layers, num_heads, 1, grid_size^2+1)
        bias_maps = torch.cat(
            [zeros_2, bias_maps], dim=-2
        )  # (n_layers, num_heads, grid_size^2+1, grid_size^2+1)

        for i, encoder in enumerate(self.encoder):
            att = encoder.mutiattention.attention
            for j , att0 in enumerate(att):
                att0.set_bias_map(bias_maps[i][j])

        self.to(self.device)

    def forward_features(self, x):
        x = self.patchembedding(x)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = self.encoder(x)
        x = self.classfier(x)
        return x




def create_lh_layer_tensor(grid_size, direction, num_heads, distance="sqrt", fill_with="inf"):
    # calculate slopes
    head_slopes = [0.25, 0.0625, 0.015625, 0.0039]

    # create list of grid coordinates
    grid_y, grid_x = grid_size
    num_patch = grid_x * grid_y
    grid_locs = np.meshgrid(range(grid_x), range(grid_y))
    coords = np.stack(grid_locs, axis=-1).reshape((-1, 2))

    # create mask for direction types
    compare_x = np.expand_dims(np.arange(grid_x), (1, 2))
    compare_y = np.expand_dims(np.arange(grid_y), (1, 2))
    direction_mask = {  # means the direction we are looking
        "right": np.less(grid_locs[0], compare_x)[coords[:, 0]],
        "left": np.greater(grid_locs[0], compare_x)[coords[:, 0]],
        "up": np.greater(grid_locs[1], compare_y)[coords[:, 1]],
        "down": np.less(grid_locs[1], compare_y)[coords[:, 1]],
    }

    # calculate dist matrix
    distvec = pdist(coords, "euclidean")
    m = squareform(distvec)
    lh = np.tile(m, (num_heads, 1, 1))
    if distance == "sqrt":
        lh = lh ** 0.5
    elif distance == "square":
        lh = lh ** 2
    elif distance == "linear":
        pass
    else:
        print("No distance passed!")

    # apply directions
    for h in range(num_heads):
        dir_mask = np.zeros_like(m, dtype=bool)
        for dir in direction[h]:
            dir_mask = dir_mask | direction_mask[dir].reshape((num_patch, -1))

            if fill_with == "inf":
                lh[h, dir_mask] = np.inf
            elif fill_with == "zero":
                lh[h, dir_mask] = 0
            elif fill_with == "max":
                lh[h, dir_mask] = lh[h, :].max()
            else:
                print("No fill_with passed!")

    # apply slopes to lh
    head_slopes = np.expand_dims(head_slopes, (1, 2))
    lh = head_slopes * lh
    return lh


def create_lh_bias_tensor(Interval, grid_size, n_layers, num_heads):
    global_slope = 1
    layer_slopes = np.linspace(
        Interval[0], Interval[1], n_layers
    )
    head_directions = [[["right"], ["left"], ["up"], ["down"]] for _ in range(n_layers)]

    lh = []
    for direction in head_directions:
        lh.append(
            create_lh_layer_tensor(
                grid_size, direction, num_heads
            )
        )

    lh = np.stack(lh, axis=0)

    # replace inf with max
    layer_slopes = np.expand_dims(layer_slopes, (1, 2, 3))
    lh = global_slope * layer_slopes * lh
    lh = np.nan_to_num(lh, posinf=torch.finfo(torch.bfloat16).max)
    lh = torch.tensor(lh, dtype=torch.bfloat16) * -1.0

    return lh



#数据预处理
image_size = (32,32)  #调整图像尺寸
n_channels = 1 #输入图像的channnel
d_model = 16  #每个patch转换为多少维度的向量
d_ratio = 4   #encoder中线性层过渡维度
num_heads = 4 #mutiattention的head数量
patch_size = 16
n_layers = 5 #encoder层数
classes = 10 #要识别的图片数量
batch_size = 128
max_sequence = 4 # max_sequence = (image_size[0]//patch_size)**2


transform = transforms.Compose([
    transforms.Resize(image_size),  # 调整尺寸
    transforms.ToTensor(),          # 转换为张量
    transforms.Normalize((0.1307,),(0.3081,)) #归一化到[0,1]
])

train_dataset = torchvision.datasets.MNIST(
                              root='../datasets',
                              train=True,
                              download=True,
                              transform=transform)

train_loader = DataLoader(train_dataset,
                          shuffle=True,
                         batch_size=batch_size,
                         pin_memory=True)

test_dataset = torchvision.datasets.MNIST(
                             root='../datasets',
                             train=False,
                             download=True,
                             transform=transform)

test_loader = DataLoader(test_dataset,
                        shuffle=False,
                        batch_size=batch_size)


model = LookHere(image_size=image_size,
                 d_model=d_model,
                 d_ratio=d_ratio,
                 num_heads=num_heads,
                 patch_size=patch_size,
                 n_channels=n_channels,
                 n_layers=n_layers,
                 classes=classes,
                 max_sequence=max_sequence,
                 device = device).to(device)

#定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):

        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        # print(outputs.size(),target.size())
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        print(batch_idx)
        if batch_idx % 300 == 0:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy on test set: %.1f %%' % (100 * correct / total))



if __name__ == '__main__':
    for epoch in range(30):
        start_time = datetime.now()
        train(epoch)
        test()
        end_time = datetime.now()
        time_diff = end_time - start_time
        print("time span:", time_diff)