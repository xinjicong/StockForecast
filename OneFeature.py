import json
import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.utils.data
import numpy as np


# 单个股票的日行情
with open('daily_basic', 'r') as f_obj:
	daily_basic=json.load(f_obj)

# 单个股票的每日的指标
with open('daily', 'r') as f_obj:
	daily = json.load(f_obj)

# 画出实际数据
plt.plot(daily['close'], c='blue')
X = daily['pre_close'][:408]
Y = daily['close'][:408]

# 超参数
EPOCH = 100
BATCH_SIZE = 64  # 每批数据数量
TIME_STEP = 8  # 时间序列长度
INPUT_SIZE = 1  # 8个特征
LR = 0.01  # 学习率

# 标准化数据
mean = np.mean(X)
std = np.std(X)
train_x = (X - mean) / std
mean = np.mean(Y)
std = np.std(Y)
train_y = (Y - mean) / std

# 批处理
torch_dataset = torch.utils.data.TensorDataset(torch.Tensor(train_x), torch.Tensor(train_y))
train_loader = torch.utils.data.DataLoader(
	dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True)


# ————————————————————————————————————————定义神经网络——————————————————————————————————————————
class RNN(nn.Module):
	def __init__(self):
		super(RNN, self).__init__()

		# self.bn_input = nn.BatchNorm1d(1, momentum=0.5)
		self.rnn = nn.LSTM(
			input_size=INPUT_SIZE,
			hidden_size=64,
			num_layers=1,
			batch_first=True,
		)
		self.out = nn.Linear(64, 1)

	def forward(self, x, h_state):
		# x = self.bn_input(x)
		r_out, h_state = self.rnn(x, None)
		outs = []
		for time_step in range(r_out.size(1)):
			outs.append(self.out(r_out[:, time_step, :]))
		return torch.stack(outs, dim=1), h_state


# ————————————————————————————————训练模型——————————————————————————————————————
rnn = RNN()
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.MSELoss()

h_state = None

for epoch in range(EPOCH):
	for step, (b_x, b_y) in enumerate(train_loader):
		b_x = b_x.view(-1, 8, 1)
		b_y = b_y.view(-1, 8, 1)

		output, h_state = rnn(b_x, h_state)
		loss = loss_func(output, b_y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print(step, loss)


# ———————————————————————————————测试模型————————————————————————————————
rnn.eval()
outputs1 = []
for a_day in torch.Tensor(train_x):
	a_day = a_day.view(1, 1, 1)
	output = rnn(a_day, None)
	outputs1.append(output[0].data*std+mean)

plt.plot(outputs1, c='red', linewidth=1)

outputs2 = []
test_x = daily['pre_close'][408:]
test_x = (test_x - mean) / std
for a_day in torch.Tensor(test_x):
	a_day = a_day.view(1, 1, 1)
	output = rnn(a_day, None)
	outputs2.append(output[0].data*std+mean)

plt.plot(range(408, 510), outputs2, c='yellow', linewidth=1)
plt.show()

