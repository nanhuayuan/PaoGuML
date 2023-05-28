import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from linear_regression import LinearRegression

# 读取数据
data = pd.read_csv('../data/world-happiness-report-2017.csv')

# 得到训练和测试数据
# 八成训练数据，2成测试数据
# sample()用于从DataFrame中随机选择行和列
# DataFrame.sample(n=None, frac=None, replace=False, weights=None, random_state=None, axis=None)
#     n：这是一个可选参数, 由整数值组成, 并定义生成的随机行数。
#     frac：它也是一个可选参数, 由浮点值组成, 并返回浮点值*数据帧值的长度。不能与参数n一起使用。
#     replace：由布尔值组成。如果为true, 则返回带有替换的样本。替换的默认值为false。
#     权重：它也是一个可选参数, 由类似于str或ndarray的参数组成。默认值”无”将导致相等的概率加权。
#     如果正在通过系列赛；它将与索引上的目标对象对齐。在采样对象中找不到的权重索引值将被忽略, 而在采样对象中没有权重的索引值将被分配零权重。
#     如果在轴= 0时正在传递DataFrame, 则返回0。它将接受列的名称。
#     如果权重是系列；然后, 权重必须与被采样轴的长度相同。
#     如果权重不等于1；它将被标准化为1的总和。
#     权重列中的缺失值被视为零。
#     权重栏中不允许无穷大。
#     random_state：它也是一个可选参数, 由整数或numpy.random.RandomState组成。如果值为int, 则为随机数生成器或numpy RandomState对象设置种子。
#     axis：它也是由整数或字符串值组成的可选参数。 0或”行”和1或”列”。
train_data = data.sample(frac = 0.8)
test_data = data.drop(train_data.index)

# 选择特征
input_param_name = 'Economy..GDP.per.Capita.'
output_param_name = 'Happiness.Score'

# 标签
x_train = train_data[[input_param_name]].values
y_train = train_data[[output_param_name]].values

x_test = test_data[input_param_name].values
y_test = test_data[output_param_name].values

plt.scatter(x_train,y_train,label='Train data')
plt.scatter(x_test,y_test,label='test data')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('Happy')
plt.legend()
plt.show()

# 迭代次数
num_iterations = 500
# 学习率
learning_rate = 0.01

# 初始化，并处理一下数据
linear_regression = LinearRegression(x_train,y_train)
# 训练数据
(theta,cost_history) = linear_regression.train(learning_rate,num_iterations)

print ('开始时的损失：',cost_history[0])
print ('训练后的损失：',cost_history[-1])

plt.plot(range(num_iterations),cost_history)
plt.xlabel('Iter')
plt.ylabel('cost')
plt.title('GD')
plt.show()

predictions_num = 100
x_predictions = np.linspace(x_train.min(),x_train.max(),predictions_num).reshape(predictions_num,1)
# 预测
y_predictions = linear_regression.predict(x_predictions)

plt.scatter(x_train,y_train,label='Train data')
plt.scatter(x_test,y_test,label='test data')
plt.plot(x_predictions,y_predictions,'r',label = 'Prediction')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('Happy')
plt.legend()
plt.show()