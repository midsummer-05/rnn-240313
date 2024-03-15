import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time


# 计算评测指标
def calculate_metrics(TP, TN, FP, FN):
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)
    print("accuracy:", accuracy)
    print("precision:", precision)
    print("recall:", recall)
    print("f1_score:", f1_score)
    return accuracy, precision, recall, f1_score


# 构建全连接网络模型
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 256)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x


# 新建网络
rnn = SimpleNet()
#
print(rnn)

# 开始计时
start = time.time()


# 训练模型
def train_model(model, train_loader, optimizer, criterion, epochs):
    model.train()
    TP, TN, FP, FN = 1, 0, 0, 0
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if epoch % 200000 == 0:
                # print(outputs.flatten())
                # predict_y=torch.max(outputs,1)[1].cpu().data.numpy().squeeze()
                # print(predict_y)
                # 假设这是网络的原始输出，未经过归一化
                print(outputs.size())
                outputs = torch.tensor(outputs[:1])

                # 计算softmax概率
                probabilities = F.softmax(outputs, dim=1)

                # 获取概率最大的类别
                predicted_class = torch.argmax(probabilities).item()

                print("Raw Output:", outputs)
                print("Probabilities:", probabilities)
                print("Predicted Class:", predicted_class)
                print("Labels:", labels[0])

                probabilities_1=probabilities
                array_1 = np.array(probabilities_1)

                X = 0
                A, B, C, D = 0, 0, 0, 0

                for i in range(len(array_1)):
                    if array_1.any() >= 0.5:
                        X += 1
                if X > 0:
                    C = 1
                else:
                    D = 1
                if predicted_class == labels[0]:
                    if C == 1:
                        TP += 1
                    else:
                        TN += 1
                if predicted_class != labels[0]:
                    if D == 1:
                        FP += 1
                    else:
                        FN += 1

                calculate_metrics(TP, TN, FP, FN)





# 结束计时
end = time.time()

# 训练耗时
print('Time cost:', end - start, 's')

# 加载MNIST数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./dataset', train=True, download=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64)

# 初始化模型、优化器和损失函数
model = SimpleNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 训练模型
train_model(model, train_loader, optimizer, criterion, epochs=2)
