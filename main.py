import argparse, json
import datetime
import os
import numpy as np
import logging
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
import random

from server import *
from client import *
import models
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures

def read_dataset():
	#遍历文件中的每一行,x是数据,y是标签,注意这里LR分类时是分到-1和1的
	# 生成数据集
	np.random.seed(0)
	x = np.random.uniform(-10, 10, (1000, 2))
	y = np.where(x[:, 1] > x[:, 0], 1, -1)

	# 将数据集划分为训练集和测试集
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
	data_X, data_Y = [], []#data_X是特征,data_Y是标签
	for a in x_train:
		data_X.append(a)
	for a in y_train:
		data_Y.append(a)
	data_X = np.array(data_X)
	data_Y = np.array(data_Y)
	print("one_num: ", np.sum(data_Y==1), ", minus_one_num: ", np.sum(data_Y==-1))
	
	idx = np.arange(data_X.shape[0])
	
	#shape 返回 (num_rows, num_columns)，其中 num_rows 是数据集的行数，num_columns 是列数。
	#shape[0] 返回行数
	
	np.random.shuffle(idx)
	
	train_size = int(data_X.shape[0]*0.8)
	
	train_x = data_X[idx[:train_size]]
	train_y = data_Y[idx[:train_size]]
	
	eval_x = data_X[idx[train_size:]]
	eval_y = data_Y[idx[train_size:]]
	
	return (train_x, train_y), (eval_x, eval_y)#返回训练集和测试集的 元组
	

if __name__ == '__main__':

	with open('./utils/conf.json', 'r') as f:
		conf = json.load(f)	#读取配置文件信息
	
	
	train_datasets, eval_datasets = read_dataset()#train_datasets是(train_x, train_y)
	#train_datasets是(train_x, train_y)
	print(train_datasets[0].shape, train_datasets[1].shape)
	
	print(eval_datasets[0].shape, eval_datasets[1].shape)
	

	server = Server(conf, eval_datasets)
	clients = []
	
	train_size = train_datasets[0].shape[0]
	per_client_size = int(train_size/conf["no_models"])#每个客户端的数据量:模拟嘛,假设有no_models个客户端
	for c in range(conf["no_models"]):
		clients.append(Client(conf, Server.public_key, server.global_model.encrypt_weights, train_datasets[0][c*per_client_size: (c+1)*per_client_size], train_datasets[1][c*per_client_size: (c+1)*per_client_size]))

	for e in range(conf["global_epochs"]):
		
		server.global_model.encrypt_weights = models.encrypt_vector(Server.public_key, models.decrypt_vector(Server.private_key, server.global_model.encrypt_weights))
		#定期(这里是每个epoch)对全局模型的参数先解密,再加密,防止发生溢出,这一行不能删
		candidates = random.sample(clients, conf["k"])#随机选取k=2个客户端
		
		weight_accumulator = [Server.public_key.encrypt(0.0)] * (conf["feature_num"]+1)
		#conf["feature_num"]+1是权重参数数量,conf["feature_num"]是特征项的系数,1个常数项
		
		for c in candidates:	
			#print(models.decrypt_vector(Server.private_key, server.global_model.encrypt_weights))
			diff = c.local_train(server.global_model.encrypt_weights)
			
			print("weights.shape= ", len(server.global_model.encrypt_weights))#weight是list

			for i in range(len(weight_accumulator)):
				weight_accumulator[i] = weight_accumulator[i] + diff[i]
			
		server.model_aggregate(weight_accumulator)#模型聚合
		
		acc = server.model_eval()#模型评估
			
		print("Epoch %d, acc: %f\n" % (e, acc))	

	

# 从模型中提取权重和偏置
with torch.no_grad():
    torch.save(server.global_model.weights, './parameters.pth')
    print("server.global_model.weights= ", server.global_model.weights)
    

    #weights = server.global_model.weights.squeeze().numpy()
    #bias = server.global_model.linear.bias.item()
params_list = torch.load('parameters.pth')
print(params_list)
x_train, y_train = train_datasets
x_test, y_test = eval_datasets
#决策边界
x_vals = np.linspace(-10, 10, 200)
y_vals = -(params_list[0] * x_vals + params_list[2]) / params_list[1]
# 绘制数据点和决策边界
plt.figure(figsize=(8, 6))

plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap='bwr', alpha=0.5, label='Train Data')
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap='coolwarm', alpha=0.5, marker='x', label='Test Data')
plt.plot(x_vals, y_vals, 'k--', label='Decision Boundary')

plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title('Data and Decision Boundary')
plt.show()