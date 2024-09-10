import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_spli


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from torch import optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# From sklearn's datasets download a subset of the 20 newsgroups dataset. This is a commonly used dataset for text classification tasks.
newsgroups_train = fetch_20newsgroups(subset='train', categories=['comp.graphics', 'sci.space'])
newsgroups_test = fetch_20newsgroups(subset='test', categories=['comp.graphics', 'sci.space'])

# 使用 CountVectorizer 将文本数据转换为词袋模型表示的数据
vectorizer = CountVectorizer(stop_words='english', max_features=1000)
inputs_train = vectorizer.fit_transform(newsgroups_train.data).toarray()

inputs_test = vectorizer.transform(newsgroups_test.data).toarray()

# 将输入和标签数据转换为张量。在 PyTorch 中, 数据需要以张量的形式进行处理
inputs_train = torch.tensor(inputs_train, dtype=torch.float32)
labels_train = torch.tensor(newsgroups_train.target, dtype=torch.long)
inputs_test = torch.tensor(inputs_test, dtype=torch.float32)
labels_test = torch.tensor(newsgroups_test.target, dtype=torch.long)



class BoWClassifier(nn.Module):
    def __init__(self, num_labels, vocab_size):
        super(BoWClassifier, self).__init__()
        self.hidden1 = nn.Linear(vocab_size, 128)
        # 隐藏层 2,输入为隐藏层 1 的输出,输出维度为 64
        self.hidden2 = nn.Linear(128, 64)

        self.output = nn.Linear(64, num_labels)
        self.relu = nn.ReLU()

    def forward(self, bow_vec):
        # 通过隐藏层 1，然后通过 ReLU 激活函数
        hidden1 = self.relu(self.hidden1(bow_vec))
        # 通过隐藏层 2，然后通过 ReLU 激活函数
        hidden2 = self.relu(self.hidden2(hidden1))
        # 通过输出层
        return self.output(hidden2)

# 根据训练数据和类别数初始化模型
vocab_size = inputs_train.shape[1]
num_labels = 2
model = BoWClassifier(num_labels, vocab_size)

# 使用交叉熵损失函数，这是一个常用于分类任务的损失函数
loss_function = nn.CrossEntropyLoss()

# 作为优化器，使用随机梯度下降(SGD)。优化器用于更新模型的参数
optimizer = optim.SGD(model.parameters(), lr=0.1)


# 训练模型。将数据传入模型,计算输出和损失,然后反向传播更新模型参数
for epoch in range(100):
    model.zero_grad() # 清除梯度
    out = model(inputs_train) # 使用输入数据预测输出
    loss = loss_function(out, labels_train) # 计算损失函数值
    loss.backward() # 反向传播以计算梯度
    optimizer.step() # 根据梯度更新模型参数


#评估模型。将测试数据传入模型,计算输出,然后与标签进行比较,计算准确率
out_test = model(inputs_test)
_, predicted = torch.max(out_test, 1)
correct = (predicted == labels_test).sum().item()
accuracy = correct / labels_test.size(0)
print('准确率：', accuracy) #0.9438