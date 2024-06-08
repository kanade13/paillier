import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 生成数据集
np.random.seed(0)
x = np.random.uniform(-10, 10, (1000, 2))
y = np.where(x[:, 1] > x[:, 0], 1, -1)

# 将数据集划分为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
i=0
for a in x_train:
    if i<10:
        i+=1
        print(a)
# 转换为PyTorch的tensor
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


class LinearClassifier(nn.Module):
    def __init__(self):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)

# 定义模型、损失函数和优化器
model = LinearClassifier()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 100
train_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train).squeeze()
    loss = criterion(outputs, (y_train + 1) / 2)
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())

    # 在测试集上评估模型
    model.eval()
    with torch.no_grad():
        test_outputs = model(x_test).squeeze()
        predictions = torch.round(torch.sigmoid(test_outputs))
        accuracy = (predictions == (y_test + 1) / 2).float().mean().item()
        test_accuracies.append(accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.4f}')

# 绘制训练损失和测试准确率
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy')
plt.legend()

plt.show()

# 从模型中提取权重和偏置
with torch.no_grad():
    weights = model.linear.weight.squeeze().numpy()
    bias = model.linear.bias.item()

# 计算决策边界的点
x_vals = np.linspace(-10, 10, 200)
y_vals = -(weights[0] * x_vals + bias) / weights[1]

# 绘制数据点和决策边界
plt.figure(figsize=(8, 6))

plt.scatter(x_train[:, 0].numpy(), x_train[:, 1].numpy(), c=y_train.numpy(), cmap='bwr', alpha=0.5, label='Train Data')
plt.scatter(x_test[:, 0].numpy(), x_test[:, 1].numpy(), c=y_test.numpy(), cmap='coolwarm', alpha=0.5, marker='x', label='Test Data')
plt.plot(x_vals, y_vals, 'k--', label='Decision Boundary')

plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title('Data and Decision Boundary')
plt.show()

# 打印模型的参数
print(f'Weights: {weights}')
print(f'Bias: {bias}')