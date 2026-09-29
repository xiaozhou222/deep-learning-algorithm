import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ============================================================
# 1. 生成原始数据
# ============================================================

torch.manual_seed(42)

# 生成 0~100 之间的1000个时间点
t = torch.linspace(0,100,1000)

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
raw_data = (0.1 * t + 0.5 * torch.sin(3 * t))

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

train_size = int(len(raw_data) * 0.8)

train_raw = raw_data[:train_size]
test_raw = raw_data[train_size:]

print("\n数据集划分：")
print("总数据量 =",len(raw_data))
print("训练集 =",len(train_raw))
print("测试集 =",len(test_raw))

# ============================================================
# 3. 只使用训练集计算 Min-Max 归一化参数
# ============================================================

train_min = train_raw.min()
train_max = train_raw.max()

# ============================================================
# 4. 训练集归一化
# ============================================================

train_data = (train_raw - train_min) / (train_max - train_min)

# ============================================================
# 5. 测试集归一化
# ============================================================

# 注意：
# 测试集也需要归一化
#
# 但是必须使用训练集的
# train_min 和 train_max

test_data = (test_raw - train_min) / (train_max - train_min)

print("\n归一化后的数据范围：")
print("训练集：", train_data.min().item(),"~",train_data.max().item())
print("测试集：",test_data.min().item(),"~",test_data.max().item())

# ============================================================
# 6. 构造训练样本
# ============================================================

# 用过去20个点预测下一个点
seq_len = 20

X_train = []
Y_train = []

for i in range(
    len(train_data) - seq_len
):

    # 输入：
    #
    # x(i) ~ x(i+19)

    X_train.append(
        train_data[
            i:i + seq_len
        ]
    )

    # 标签：
    #
    # x(i+20)

    Y_train.append(
        train_data[
            i + seq_len
        ]
    )

# ============================================================
# 转换成 Tensor
# ============================================================

X_train = torch.stack(X_train)
Y_train = torch.stack(Y_train)

# ============================================================
# 增加 feature 维度
# ============================================================

# 原来：
#
# [780, 20]
#
# ↓
#
# [780, 20, 1]
#
#
# 780 = 样本数量
# 20  = 时间步
# 1   = 每个时间步1个特征

X_train = X_train.unsqueeze(-1)
Y_train = Y_train.unsqueeze(-1)

print("\n训练集样本形状：")
print("X_train shape:",X_train.shape)
print("Y_train shape:",Y_train.shape)

# ============================================================
# 7. 构造测试样本
# ============================================================

# 测试开始时，需要知道测试集开始之前的20个历史数据。
#
# 所以：
#
# 训练集最后20个数据
# +
# 整个测试集
#
# 一起构成测试阶段的数据窗口。

test_input_data = torch.cat(
    [
        train_data[-seq_len:],#表示从训练集末尾开始取20个数据
        test_data
    ]
)


X_test = []
Y_test = []


for i in range(
    len(test_data)
):

    # 过去20个点
    X_test.append(
        test_input_data[
            i:i + seq_len
        ]
    )

    # 预测下一个点
    Y_test.append(
        test_input_data[
            i + seq_len
        ]
    )

# ============================================================
# 转换成 Tensor
# ============================================================

X_test = torch.stack(X_test)
Y_test = torch.stack(Y_test)

# ============================================================
# 增加 feature 维度
# ============================================================

X_test = X_test.unsqueeze(-1)
Y_test = Y_test.unsqueeze(-1)

print("\n测试集样本形状：")
print("X_test shape:",X_test.shape)
print("Y_test shape:",Y_test.shape)

# ============================================================
# 6. 自定义 LSTM Cell
#
# 这里是本次实验最核心的部分
# ============================================================

class MyLSTMCell(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size
    ):

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # ----------------------------------------------------
        # 一次性计算4个门
        #
        # 输入：
        #
        # [x_t, h_(t-1)]
        #
        # 输出：
        #
        # 4 * hidden_size
        #
        # 然后拆成：
        #
        # forget gate遗忘门
        # input gate输入门
        # candidate候选细胞状态
        # output gate输出门
        # ----------------------------------------------------

        self.linear = nn.Linear(
            input_size + hidden_size,
            4 * hidden_size
        )                               # 创建一个全连接层，把 当前输入 x_t 和 上一时刻隐藏状态 h_(t-1) 拼接后的向量，映射成 LSTM 四个门所需要的预激活值。

    def forward(
        self,
        x,
        hidden_state
    ):

        # ----------------------------------------------------
        # 上一个时间步的：
        #
        # h_(t-1)
        # C_(t-1)
        # ----------------------------------------------------

        h_prev, c_prev = hidden_state

        # ----------------------------------------------------
        # 拼接：
        #
        # [x_t, h_(t-1)]
        # ----------------------------------------------------

        combined = torch.cat([x,h_prev],dim=1)#当前输入和过去的隐藏状态拼接在一起，作为全连接层的输入

        # ----------------------------------------------------
        # 计算4组原始值
        # ----------------------------------------------------

        gates = self.linear(combined)        # 通过全连接层计算四个门的原始值64

        # ----------------------------------------------------
        # 拆成4部分
        #
        # [batch, 256]
        #
        # ↓
        #
        # 4 × [batch, 64]
        # ----------------------------------------------------

        forget_raw, \
        input_raw, \
        candidate_raw, \
        output_raw = torch.chunk(
            gates,
            4,
            dim=1
        )

        # ====================================================
        # ① Forget Gate
        #
        # f_t = sigmoid(...)
        #
        # 范围：
        #
        # 0 ~ 1
        # ====================================================

        f_t = torch.sigmoid(forget_raw)#输入的原始值经过sigmoid函数，得到遗忘门的值，范围在0到1之间，即保留多少旧记忆

        # ====================================================
        # ② Input Gate
        #
        # i_t = sigmoid(...)
        #
        # 范围：
        #
        # 0 ~ 1
        # ====================================================

        i_t = torch.sigmoid(input_raw)#输入的原始值经过sigmoid函数，得到输入门的值，范围在0到1之间，即允许多少新信息进入

        # ====================================================
        # ③ Candidate Cell State
        #
        # C~_t = tanh(...)
        #
        # 范围：
        #
        # -1 ~ 1
        # ====================================================

        candidate_t = torch.tanh(candidate_raw)#输入的原始值经过tanh函数，得到候选细胞状态的值，范围在-1到1之间，即新信息的候选值（筛选什么值得留下）

        # ====================================================
        # ④ Output Gate
        #
        # o_t = sigmoid(...)
        #
        # 范围：
        #
        # 0 ~ 1
        # ====================================================

        o_t = torch.sigmoid(output_raw)#输入的原始值经过sigmoid函数，得到输出门的值，范围在0到1之间，即允许多少信息输出

        # ====================================================
        # ⑤ 更新 Cell State
        #
        # C_t =
        #
        # f_t * C_(t-1)
        # +
        # i_t * C~_t
        # ====================================================

        c_t = (f_t * c_prev + i_t * candidate_t)#更新细胞状态，遗忘旧信息并加入新信息

        # ====================================================
        # ⑥ 更新 Hidden State
        #
        # h_t =
        #
        # o_t * tanh(C_t)
        # ====================================================

        h_t = (o_t * torch.tanh(c_t))

        # ====================================================
        # 返回：
        #
        # h_t
        # C_t
        #
        # 以及4个门
        # ====================================================

        return (h_t,c_t,f_t,i_t,candidate_t,o_t)

# ============================================================
# 8. 定义 LSTM 模型
# ============================================================

class LSTMModel(nn.Module):
    
    def __init__(
        self,
        input_size=1,
        hidden_size=32,
        num_layers=2
    ):

        super().__init__()

        # ----------------------------------------------------
        # 保存模型参数
        # ----------------------------------------------------

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # ----------------------------------------------------
        # 第一层 LSTMCell
        #
        # 输入：
        # [batch_size, 1]
        #
        # 输出：
        # h1：
        # [batch_size, 64]
        #
        # c1：
        # [batch_size, 64]
        # ----------------------------------------------------

        self.lstm1 = MyLSTMCell(
            input_size,
            hidden_size
        )

        # ----------------------------------------------------
        # 第二层 LSTMCell
        #
        # 第一层输出 h1
        # 作为第二层输入
        #
        # [batch_size, 64]
        #
        # ↓
        #
        # [batch_size, 64]
        # ----------------------------------------------------

        self.lstm2 = MyLSTMCell(
            hidden_size,
            hidden_size
        )

        # ----------------------------------------------------
        # 全连接层
        #
        # [batch_size, 64]
        #
        # ↓
        #
        # [batch_size, 1]
        # ----------------------------------------------------

        self.fc = nn.Linear(
            hidden_size,
            1
        )


    def forward(
        self,
        x,
        record_state=False
    ):

        # ====================================================
        # x：
        #
        # [batch_size, seq_len, input_size]
        #
        # 当前：
        #
        # [batch_size, 20, 1]
        # ====================================================

        batch_size = x.size(0)

        device = x.device

        # ====================================================
        # 初始化第一层 Hidden State
        #
        # h1：
        # [batch_size, 64]
        # ====================================================

        h1 = torch.zeros(
            batch_size,
            self.hidden_size,
            device=device
        )

        # ====================================================
        # 初始化第一层 Cell State
        #
        # c1：
        # [batch_size, 64]
        # ====================================================

        c1 = torch.zeros(
            batch_size,
            self.hidden_size,
            device=device
        )

        # ====================================================
        # 初始化第二层 Hidden State
        # ====================================================

        h2 = torch.zeros(
            batch_size,
            self.hidden_size,
            device=device
        )

        # ====================================================
        # 初始化第二层 Cell State
        # ====================================================

        c2 = torch.zeros(
            batch_size,
            self.hidden_size,
            device=device
        )

        # ====================================================
        # 用来保存每一个时间步的状态
        # ====================================================

        hidden_states = []
        cell_states = []

        forget_gates = []
        input_gates = []
        candidate_states = []
        output_gates = []
        
        # ====================================================
        # 一个时间步一个时间步运行 LSTM
        # ====================================================

        for time_step in range(
            x.size(1)
        ):


            # ------------------------------------------------
            # 当前时间步输入
            #
            # x[:, 0, :]
            # x[:, 1, :]
            # ...
            # x[:, 19, :]
            #
            # shape：
            #
            # [batch_size, 1]
            # ------------------------------------------------

            xt = x[
                :,
                time_step,
                :
            ]


            # ------------------------------------------------
            # 第一层 LSTM
            #
            # 输入：
            #
            # xt
            # h1
            # c1
            #
            # 输出：
            #
            # h1
            # c1
            # ------------------------------------------------

            (h1, c1 ,f1,i1,candidata1,o1) = self.lstm1(xt,(h1, c1))

            # ------------------------------------------------
            # 第二层 LSTM
            #
            # 第一层的 h1
            # 作为第二层输入
            # ------------------------------------------------

            (h2, c2,f2,i2,candidate2,o2) = self.lstm2(h1,(h2, c2))

            # ------------------------------------------------
            # 记录第二层状态
            #
            # 因为我们主要观察第二层
            # ------------------------------------------------

            hidden_states.append(h2)

            cell_states.append(c2)

            forget_gates.append(f2)

            input_gates.append(i2)

            candidate_states.append(candidate2)

            output_gates.append(o2)

        # ====================================================
        # 把20个时间步的数据堆叠起来
        #
        # 原始：
        #
        # 20个：
        #
        # [batch_size, 64]
        #
        #
        # stack之后：
        #
        # [20, batch_size, 64]
        # ====================================================

        hidden_states = torch.stack(
            hidden_states,
            dim=1
        )

        cell_states = torch.stack(
            cell_states,
            dim=1
        )

        forget_gates = torch.stack(
            forget_gates,
            dim=1
        )

        input_gates = torch.stack(
            input_gates,
            dim=1
        )

        candidate_states = torch.stack(
            candidate_states,
            dim=1
        )

        output_gates = torch.stack(
            output_gates,
            dim=1
        )

        # ====================================================
        # 取最后一个时间步
        #
        # hidden_states：
        #
        # [batch_size, 20, 64]
        #
        # ↓
        #
        # [batch_size, 64]
        # ====================================================

        out = hidden_states[:,-1,:]

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

        # ====================================================
        # 如果需要观察内部状态
        # 就把状态一起返回
        # ====================================================

        if record_state:
            return (
                out,
                hidden_states,
                cell_states,
                forget_gates,
                input_gates,
                candidate_states,
                output_gates
            )
        else:
            return out

# ============================================================
# 9. 创建模型
# ============================================================

model = LSTMModel(
    input_size=1,
    hidden_size=32,
    num_layers=2
)

print("\n模型结构：")
print(model)

# ============================================================
# 10. 定义损失函数和优化器
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

# ============================================================
# 11. 训练模型
# ============================================================

epochs = 300

loss_list = []

for epoch in range(
    epochs
):

    # --------------------------------------------------------
    # ① 前向传播
    # --------------------------------------------------------

    prediction = model(
        X_train
    )

    # --------------------------------------------------------
    # ② 计算Loss
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
    # ④ 反向传播
    # --------------------------------------------------------

    loss.backward()

    # --------------------------------------------------------
    # ⑤ 更新参数
    # --------------------------------------------------------

    optimizer.step()

    # --------------------------------------------------------
    # 保存Loss
    # --------------------------------------------------------

    loss_list.append(
        loss.item()
    )

    # --------------------------------------------------------
    # 输出训练过程
    # --------------------------------------------------------

    if (
        epoch + 1
    ) % 10 == 0:

        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Training Loss: {loss.item():.6f}"
        )


# ============================================================
# 12. 测试集预测
# ============================================================

model.eval()

with torch.no_grad():

    prediction = model(
        X_test
    )

# ============================================================
# 13. 反归一化
# ============================================================

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
# 14. 计算测试集 MSE
# ============================================================

test_mse = torch.mean(
    (
        prediction_original
        - Y_test_original
    ) ** 2
)

# ============================================================
# 15. 计算测试集 RMSE
# ============================================================

test_rmse = torch.sqrt(
    test_mse
)

print("\n==============================")
print("测试集评价结果")
print("==============================")

print(
    "Test Original Scale MSE:",
    test_mse.item()
)

print(
    "Test Original Scale RMSE:",
    test_rmse.item()
)

# ============================================================
# 16. 绘制原始数据
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    t.numpy(),
    raw_data.numpy(),
    label="Original Data"
)

# 训练集 / 测试集分界线
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
# 17. 绘制测试集预测结果
# ============================================================

plt.figure(
    figsize=(10, 5)
)

# 测试集时间

test_t = t[
    train_size:
]

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
# 18. 绘制训练 Loss
# ============================================================

plt.figure(
    figsize=(8, 4)
)

plt.plot(
    loss_list
)

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss")

plt.show()

# ============================================================
# 19. 获取一个测试样本
#
# 用于观察 LSTM 内部状态
# ============================================================

sample = X_test[
    0:1
]

with torch.no_grad():

    (
        prediction_sample,
        hidden_states,
        cell_states,
        forget_gates,
        input_gates,
        candidate_states,
        output_gates

    ) = model(
        sample,
        record_state=True
    )


# ============================================================
# 18. 打印所有状态的 Shape
# ============================================================

print("\n==============================")
print("LSTM内部状态 Shape")
print("==============================")

print(
    "Hidden State:",
    hidden_states.shape
)

print(
    "Cell State:",
    cell_states.shape
)

print(
    "Forget Gate:",
    forget_gates.shape
)

print(
    "Input Gate:",
    input_gates.shape
)

print(
    "Candidate:",
    candidate_states.shape
)

print(
    "Output Gate:",
    output_gates.shape
)

# ============================================================
# 20. 提取第一个样本
#
# hidden_states：
#
# [1, 20, 64]
#
# ↓
#
# [20, 64]
# ============================================================

hidden = hidden_states[
    0
].numpy()

cell = cell_states[
    0
].numpy()

forget = forget_gates[
    0
].numpy()

input_gate = input_gates[
    0
].numpy()

candidate = candidate_states[
    0
].numpy()

output_gate = output_gates[
    0
].numpy()

# ============================================================
# 21. 绘制当前测试样本输入
# ============================================================

sample_input = sample[
    0,
    :,
    0
].numpy()

plt.figure(
    figsize=(10, 4)
)

plt.plot(
    range(1, seq_len + 1),
    sample_input,
    marker="o"
)

plt.xlabel("Time Step")
plt.ylabel("Normalized Value")
plt.title("Input Sequence")

plt.grid(True)
plt.show()


# ============================================================
# 22. Hidden State 热力图
# ============================================================

plt.figure(
    figsize=(12, 6)
)


plt.imshow(
    hidden.T,
    aspect="auto"
)


plt.colorbar(label="Hidden State Value")
plt.xlabel("Time Step")
plt.ylabel( "Hidden Dimension")
plt.title("LSTM Hidden State")

plt.show()

# ============================================================
# 23. Cell State 热力图
# ============================================================

plt.figure(
    figsize=(12, 6)
)


plt.imshow(
    cell.T,
    aspect="auto"
)


plt.colorbar(label="Cell State Value")
plt.xlabel("Time Step")
plt.ylabel("Cell Dimension")
plt.title("LSTM Cell State")

plt.show()

# ============================================================
# 24. Forget Gate 热力图
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.imshow(
    forget.T,
    aspect="auto",
    vmin=0,
    vmax=1
)

plt.colorbar(
    label="Forget Gate Value"
)

plt.xlabel("Time Step")
plt.ylabel("Hidden Dimension")

plt.title(
    "LSTM Forget Gate"
)

plt.show()

# ============================================================
# 24. Input Gate 热力图
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.imshow(
    input_gate.T,
    aspect="auto",
    vmin=0,
    vmax=1
)

plt.colorbar(
    label="Input Gate Value"
)

plt.xlabel("Time Step")
plt.ylabel("Hidden Dimension")

plt.title(
    "LSTM Input Gate"
)

plt.show()


# ============================================================
# 25. Candidate Cell State 热力图
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.imshow(
    candidate.T,
    aspect="auto",
    vmin=-1,
    vmax=1
)

plt.colorbar(
    label="Candidate Value"
)

plt.xlabel("Time Step")
plt.ylabel("Hidden Dimension")

plt.title(
    "LSTM Candidate Cell State"
)

plt.show()


# ============================================================
# 26. Output Gate 热力图
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.imshow(
    output_gate.T,
    aspect="auto",
    vmin=0,
    vmax=1
)

plt.colorbar(
    label="Output Gate Value"
)

plt.xlabel("Time Step")
plt.ylabel("Hidden Dimension")

plt.title(
    "LSTM Output Gate"
)

plt.show()

# ============================================================
# 27. 第1个神经元：四个门
# ============================================================

neuron = 0

time_steps = range(
    1,
    seq_len + 1
)


plt.figure(
    figsize=(10, 5)
)

plt.plot(
    time_steps,
    forget[:, neuron],
    marker="o",
    label="Forget Gate"
)

plt.plot(
    time_steps,
    input_gate[:, neuron],
    marker="o",
    label="Input Gate"
)

plt.plot(
    time_steps,
    output_gate[:, neuron],
    marker="o",
    label="Output Gate"
)

plt.plot(
    time_steps,
    candidate[:, neuron],
    marker="o",
    label="Candidate"
)

plt.xlabel("Time Step")

plt.ylabel("Gate / Candidate Value")

plt.title(
    "LSTM Gates of Neuron 1"
)

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# 28. 第1个神经元：
#     Cell State + Hidden State
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    time_steps,
    cell[:, neuron],
    marker="o",
    label="Cell State"
)

plt.plot(
    time_steps,
    hidden[:, neuron],
    marker="o",
    label="Hidden State"
)

plt.xlabel("Time Step")

plt.ylabel("State Value")

plt.title(
    "Cell State vs Hidden State of Neuron 1"
)

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# 29. 打印第1个神经元的完整状态
# ============================================================

print("\n==============================")
print("Neuron 1 的 LSTM 内部状态")
print("==============================")


for i in range(seq_len):

    print(
        f"Time Step {i + 1:2d} | "
        f"Forget={forget[i, neuron]:.4f} | "
        f"Input={input_gate[i, neuron]:.4f} | "
        f"Candidate={candidate[i, neuron]:.4f} | "
        f"Output={output_gate[i, neuron]:.4f} | "
        f"Cell={cell[i, neuron]:.4f} | "
        f"Hidden={hidden[i, neuron]:.4f}"
    )
    
# ============================================================
# 计算预测偏差
# ============================================================

mean_error = torch.mean(
    prediction_original - Y_test_original
)

mae = torch.mean(
    torch.abs(
        prediction_original - Y_test_original
    )
)

print("Mean Error:", mean_error.item())
print("MAE:", mae.item())

pred_max = prediction_original.max()
pred_min = prediction_original.min()

true_max = Y_test_original.max()
true_min = Y_test_original.min()

pred_amplitude = (pred_max - pred_min) / 2
true_amplitude = (true_max - true_min) / 2

amplitude_ratio = (
    pred_amplitude / true_amplitude
)

print("True Amplitude:", true_amplitude.item())
print("Prediction Amplitude:", pred_amplitude.item())
print("Amplitude Ratio:",amplitude_ratio.item())