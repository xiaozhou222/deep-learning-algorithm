import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ============================================================
# 1. 生成原始数据
# ============================================================

torch.manual_seed(42)  # 设置随机种子，保证每次运行结果基本一致

# 生成 0~100 之间的1000个时间点
t = torch.linspace(0, 100, 1000)

# ============================================================
# 选择要预测的函数
# ============================================================

# ① 正弦函数
# raw_data = torch.sin(t)

# ② 余弦函数
# raw_data = torch.cos(t)

# ③ 多频率正弦函数
# raw_data = (
#     torch.sin(t)
#     + 0.5 * torch.cos(3 * t)
#     + 0.2 * torch.sin(5 * t)
# )

# ④ 线性 + 正弦
raw_data = 0.1 * t + 0.5 * torch.sin(3 * t)

# ⑤ 二次函数
# raw_data = t ** 2

# ⑥ 指数函数
# raw_data = torch.exp(0.05 * t)


print("原始数据范围：")
print("最小值 =", raw_data.min().item())
print("最大值 =", raw_data.max().item())


# ============================================================
# 2. 划分训练集和测试集
# ============================================================

# 使用80%的数据进行训练
train_size = int(len(raw_data) * 0.8)

train_raw = raw_data[:train_size]
test_raw = raw_data[train_size:]


print("\n数据集划分：")
print("总数据量 =", len(raw_data))
print("训练集 =", len(train_raw))
print("测试集 =", len(test_raw))


# ============================================================
# 3. 只使用训练集计算 Min-Max 归一化参数
# ============================================================

# 注意：
# train_min 和 train_max 只能由训练集计算
# 不能使用整个数据集，否则会产生数据泄漏

train_min = train_raw.min()
train_max = train_raw.max()


# ============================================================
# 4. 训练集归一化
# ============================================================

train_data = (
    train_raw - train_min
) / (
    train_max - train_min
)


# ============================================================
# 5. 测试集归一化
# ============================================================

# 注意：
# 测试集也需要归一化
# 但是必须使用训练集的 train_min 和 train_max

test_data = (
    test_raw - train_min
) / (
    train_max - train_min
)


print("\n归一化后的数据范围：")

print(
    "训练集：",
    train_data.min().item(),
    "~",
    train_data.max().item()
)

print(
    "测试集：",
    test_data.min().item(),
    "~",
    test_data.max().item()
)


# ============================================================
# 6. 构造训练样本
# ============================================================

seq_len = 20

# 用过去20个点预测下一个点
#
# 例如：
#
# x1 x2 x3 ... x20 → x21
# x2 x3 x4 ... x21 → x22
# x3 x4 x5 ... x22 → x23

X_train = []
Y_train = []


for i in range(len(train_data) - seq_len):

    X_train.append(
        train_data[i:i + seq_len]
    )

    Y_train.append(
        train_data[i + seq_len]
    )


# 转换成 Tensor
X_train = torch.stack(X_train)
Y_train = torch.stack(Y_train)


# 增加 feature 维度
#
# 原来：
# X_train = [780, 20]
#
# 现在：
# X_train = [780, 20, 1]
#
# 含义：
# 780 = 样本数量
# 20  = 每个样本包含20个时间步
# 1   = 每个时间步只有1个特征

X_train = X_train.unsqueeze(-1)
Y_train = Y_train.unsqueeze(-1)


print("\n训练集样本形状：")
print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)


# ============================================================
# 7. 构造测试样本
# ============================================================

# 测试开始的时候，模型需要知道测试集开始之前的20个历史数据。
#
# 例如：
#
# 训练集             测试集
# ----------------|----------------
# ... 780 781 ... | 800 801 802 ...
#       ↑
#   历史窗口
#
# 第一个测试数据需要：
#
# 过去20个训练数据 → 第一个测试数据
#
# 所以取训练集最后20个点作为测试阶段的历史窗口。

test_input_data = torch.cat(
    [
        train_data[-seq_len:],
        test_data
    ]
)


X_test = []
Y_test = []


for i in range(len(test_data)):

    # 使用过去20个点
    X_test.append(
        test_input_data[i:i + seq_len]
    )

    # 预测下一个点
    Y_test.append(
        test_input_data[i + seq_len]
    )


# 转换成 Tensor
X_test = torch.stack(X_test)
Y_test = torch.stack(Y_test)


# 增加 feature 维度

X_test = X_test.unsqueeze(-1)
Y_test = Y_test.unsqueeze(-1)


print("\n测试集样本形状：")
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)

# ============================================================
# LSTM内部状态可视化模型
# ============================================================

class LSTMStateModel(nn.Module):

    def __init__(self):

        super().__init__()

        # 第一层 LSTMCell
        self.lstm1 = nn.LSTMCell(
            input_size=1,
            hidden_size=64
        )

        # 第二层 LSTMCell
        self.lstm2 = nn.LSTMCell(
            input_size=64,
            hidden_size=64
        )

        # 全连接层
        self.fc = nn.Linear(
            64,
            1
        )


    def forward(self, x, record_state=False):

        # ====================================================
        # x:
        #
        # [batch_size, seq_len, 1]
        #
        # ====================================================

        batch_size = x.size(0)

        device = x.device


        # ====================================================
        # 初始化第一层状态
        #
        # h1：第一层 hidden state
        # c1：第一层 cell state
        # ====================================================

        h1 = torch.zeros(
            batch_size,
            64,
            device=device
        )

        c1 = torch.zeros(
            batch_size,
            64,
            device=device
        )


        # ====================================================
        # 初始化第二层状态
        # ====================================================

        h2 = torch.zeros(
            batch_size,
            64,
            device=device
        )

        c2 = torch.zeros(
            batch_size,
            64,
            device=device
        )


        # ====================================================
        # 用来保存每一个时间步的状态
        # ====================================================

        hidden_states = []

        cell_states = []


        # ====================================================
        # 一个时间步一个时间步处理
        # ====================================================

        for t in range(x.size(1)):

            # ------------------------------------------------
            # 当前时间步输入
            # ------------------------------------------------

            xt = x[:, t, :]


            # ------------------------------------------------
            # 第一层 LSTM
            # ------------------------------------------------

            h1, c1 = self.lstm1(
                xt,
                (h1, c1)
            )


            # ------------------------------------------------
            # 第二层 LSTM
            # ------------------------------------------------

            h2, c2 = self.lstm2(
                h1,
                (h2, c2)
            )


            # ------------------------------------------------
            # 保存第二层状态
            # ------------------------------------------------

            hidden_states.append(h2)

            cell_states.append(c2)


        # ====================================================
        # [seq_len, batch_size, hidden_size]
        #
        # ↓
        #
        # [batch_size, seq_len, hidden_size]
        # ====================================================

        hidden_states = torch.stack(
            hidden_states,
            dim=1
        )

        cell_states = torch.stack(
            cell_states,
            dim=1
        )


        # ====================================================
        # 最后一个时间步
        # ====================================================

        out = hidden_states[:, -1, :]


        # ====================================================
        # 全连接层
        # ====================================================

        out = self.fc(out)


        if record_state:

            return (
                out,
                hidden_states,
                cell_states
            )

        else:

            return out

# ============================================================
# 8. 定义 LSTM 模型
# ============================================================

class LSTMModel(nn.Module):

    def __init__(self):

        super().__init__()

        # ====================================================
        # LSTM层
        # ====================================================

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        # ====================================================
        # 全连接层
        # ====================================================

        self.fc = nn.Linear(
            64,
            1
        )


    def forward(self, x):

        # ====================================================
        # LSTM前向传播
        #
        # 输入：
        # [batch_size, seq_len, input_size]
        #
        # 输出：
        # output：
        # [batch_size, seq_len, hidden_size]
        #
        # hidden：
        # [num_layers, batch_size, hidden_size]
        #
        # cell：
        # [num_layers, batch_size, hidden_size]
        # ====================================================

        output, (hidden, cell) = self.lstm(x)#hidden,cell为lstm的隐藏状态和细胞状态

        # ====================================================
        # 只取最后一个时间步
        #
        # [batch_size, seq_len, hidden_size]
        #
        # ↓
        #
        # [batch_size, hidden_size]
        # ====================================================

        out = output[:, -1, :]


        # ====================================================
        # 全连接层
        #
        # [batch_size, 64]
        #
        # ↓
        #
        # [batch_size, 1]
        # ====================================================

        out = self.fc(out)


        return out


# 创建模型

model = LSTMModel()


print("\n模型结构：")
print(model)

# ============================================================
# 9. 定义损失函数和优化器
# ============================================================

criterion = nn.MSELoss()


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)


# ============================================================
# 10. 训练模型
# ============================================================

epochs = 200

loss_list = []


for epoch in range(epochs):


    # --------------------------------------------------------
    # ① 前向传播
    # --------------------------------------------------------

    prediction = model(X_train)


    # --------------------------------------------------------
    # ② 计算训练集 Loss
    # --------------------------------------------------------

    loss = criterion(
        prediction,
        Y_train
    )


    # --------------------------------------------------------
    # ③ 梯度清零
    # --------------------------------------------------------

    optimizer.zero_grad()


    # --------------------------------------------------------
    # ④ BPTT + 反向传播
    # --------------------------------------------------------

    loss.backward()


    # --------------------------------------------------------
    # ⑤ 梯度下降，更新参数
    # --------------------------------------------------------

    optimizer.step()


    # 保存Loss

    loss_list.append(
        loss.item()
    )


    # 输出训练过程

    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Training Loss: {loss.item():.6f}"
        )


# ============================================================
# 11. 测试集预测
# ============================================================

# 切换到评估模式

model.eval()


# 测试时不需要计算梯度

with torch.no_grad():

    prediction = model(X_test)


# ============================================================
# 12. 反归一化
# ============================================================

# LSTM预测出来的是归一化之后的数据。
#
# 需要恢复成原来的真实数值：
#
# y = y_norm × (max - min) + min
#
# 注意：
# 这里仍然使用训练集的 train_min 和 train_max。


prediction_original = (
    prediction
    * (train_max - train_min)
    + train_min
)


Y_test_original = (
    Y_test
    * (train_max - train_min)
    + train_min
)


# ============================================================
# 13. 计算测试集原始尺度 MSE
# ============================================================

test_mse = torch.mean(
    (
        prediction_original
        - Y_test_original
    ) ** 2
)


print("\n==============================")
print("测试集评价结果")
print("==============================")

print(
    "Test Original Scale MSE:",
    test_mse.item()
)


# ============================================================
# 14. 计算测试集原始尺度 RMSE
# ============================================================

test_rmse = torch.sqrt(
    test_mse
)


print(
    "Test Original Scale RMSE:",
    test_rmse.item()
)


# ============================================================
# 15. 绘制原始数据
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    t.numpy(),
    raw_data.numpy(),
    label="Original Data"
)

# 标记训练集和测试集分界线

plt.axvline(
    t[train_size].item(),
    linestyle="--",
    label="Train/Test Split"
)

plt.xlabel("Time")

plt.ylabel("Original Value")

plt.title("Original Time Series")

plt.legend()

plt.show()


# ============================================================
# 16. 绘制测试集预测结果
# ============================================================

plt.figure(figsize=(10, 5))


# 测试集对应的时间

test_t = t[train_size:]


plt.plot(
    test_t.numpy(),
    Y_test_original.numpy(),
    label="True"
)


plt.plot(
    test_t.numpy(),
    prediction_original.numpy(),
    label="Prediction"
)


plt.xlabel("Time")

plt.ylabel("Original Value")

plt.title("LSTM Test Prediction")

plt.legend()

plt.show()


# ============================================================
# 17. 绘制训练 Loss
# ============================================================

plt.figure(figsize=(8, 4))


plt.plot(
    loss_list
)


plt.xlabel("Epoch")

plt.ylabel("Training Loss")

plt.title("Training Loss")

plt.show()