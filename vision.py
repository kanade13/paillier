import numpy as np
import torch
import torch.nn as nn
params_list = torch.load('parameters.pth')
print(params_list)

#决策边界
x_vals = np.linspace(-10, 10, 200)
y_vals = -(params_list[0] * x_vals + params_list[2]) / params_list[1]
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