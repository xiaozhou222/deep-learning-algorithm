import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math


# =========================================================
# 1. 固定随机种子
# =========================================================
torch.manual_seed(42)

#宏定义一些重要参数，方便修改
t_lr=0.001
t_epochs=300
t_seq_len=20

t_input_size=1        # 输入特征维度
t_d_model=64          # Transformer模型的隐藏维度
t_nhead=4             # 多头注意力机制的头数
t_num_layers=2        # Transformer Encoder Layer的层数
t_dim_feedforward=128 # Transformer中前馈神经网络的隐藏层维度
t_dropout=0.1         # Transformer中dropout的比例

# =========================================================
# 2. 生成原始时间序列
# =========================================================
t = torch.linspace(0, 100, 1000)

raw_data = 0.1 * t + 0.5 * torch.sin(3 * t)

print("原始数据形状：", raw_data.shape)


# =========================================================
# 3. 先划分原始时间序列
# =========================================================
# 前800个点用于训练
# 后200个点用于测试
# =========================================================
train_size = 800
train_raw = raw_data[:train_size]
test_raw = raw_data[train_size:]

print("训练原始数据形状：", train_raw.shape)
print("测试原始数据形状：", test_raw.shape)


# =========================================================
# 4. 只使用训练集计算 mean 和 std
# =========================================================
train_t = t[:train_size]
test_t = t[train_size:]

# 构造最小二乘矩阵 [t, 1]
A = torch.stack([train_t, torch.ones_like(train_t)], dim=1)
b = train_raw

# 求解线性回归 y = a*t + b
coeff = torch.linalg.lstsq(A, b).solution
a, b_val = coeff[0], coeff[1]

print(f"\n拟合趋势: y = {a.item():.4f} * t + {b_val.item():.4f}")

# 计算整条曲线的趋势
trend_all = a * t + b_val
residual_all = raw_data - trend_all

# 划分残差
train_res = residual_all[:train_size]
test_res  = residual_all[train_size:]


# =========================================================
# 5. 使用训练集的 mean/std 进行标准化
# =========================================================
train_mean = train_res.mean()
train_std  = train_res.std()

print("\n训练集残差 mean:", train_mean.item())
print("训练集残差 std :", train_std.item())

train_data = (train_res - train_mean) / train_std
test_data  = (test_res  - train_mean) / train_std

# =========================================================
# 6. 构造滑动窗口函数
# =========================================================
def create_sequences(data, seq_len):

    X = []
    Y = []

    for i in range(len(data) - seq_len):
        #输入连续20个时间点
        X.append(
            data[i:i + seq_len]
        )
        #第21个时间点作为预测目标
        Y.append(
            data[i + seq_len]
        )

    X = torch.stack(X)
    Y = torch.stack(Y)

    # 增加 feature 维度
    #
    # [样本数, seq_len]
    #
    # 变成：
    #
    # [样本数, seq_len, 1]
    X = X.unsqueeze(-1)
    Y = Y.unsqueeze(-1)

    return X, Y


# =========================================================
# 7. 设置序列长度
# =========================================================
seq_len = t_seq_len

# =========================================================
# 8. 分别构造训练集和测试集
# =========================================================
X_train, Y_train = create_sequences(
    train_data,
    seq_len
)

X_test, Y_test = create_sequences(
    test_data,
    seq_len
)


print("\n训练集：")
print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)

print("\n测试集：")
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)


# =========================================================
# 9. Positional Encoding
# =========================================================
class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_len=500         # 最大序列长度
    ):
        super().__init__()

        # 创建位置编码矩阵
        #
        # [max_len, d_model]
        pe = torch.zeros(
            max_len,
            d_model
        )

        # position，[max_len, 1]
        position = torch.arange(
            0,
            max_len,
            dtype=torch.float
        ).unsqueeze(1)

        # 计算分母,div_term，[d_model/2]，用于控制不同维度的频率
        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2
            ).float()
            * (
                -torch.log(
                    torch.tensor(10000.0)
                )
                / d_model
            )
        )

        # 偶数维使用 sin
        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        # 奇数维使用 cos
        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        # 增加 batch 维，[1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # 注册为 buffer，不作为模型参数，但会随模型一起保存和加载
        self.register_buffer(
            "pe",
            pe
        )

    def forward(self, x):

        # x:
        # [batch, seq_len, d_model]

        x = x + self.pe[:, :x.size(1), :]

        return x


# =========================================================
# 10. Transformer Model
# =========================================================
class TransformerModel(nn.Module):

    def __init__(
        self,
        input_size=t_input_size,           # 输入特征维度
        d_model=t_d_model,                 # Transformer模型的隐藏维度
        nhead=t_nhead,                     # 多头注意力机制的头数
        num_layers=t_num_layers,           # Transformer Encoder Layer的层数
        dim_feedforward=t_dim_feedforward, # Transformer中前馈神经网络的隐藏层维度
        dropout=t_dropout                  # Transformer中dropout的比例
    ):

        super().__init__()
        self.d_model = d_model
        
        # -------------------------------------------------
        # 输入映射
        # 1维 → 64维
        # -------------------------------------------------
        self.input_projection = nn.Linear(
            input_size,
            d_model
        )                    # 输入映射层，[B,20,1]->[B,20,64]，B为batch_size，20为序列长度，1为输入特征维度，64为Transformer模型的隐藏维度

        # -------------------------------------------------
        # 位置编码
        # -------------------------------------------------
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=seq_len
        )

        # -------------------------------------------------
        # Transformer Encoder Layer
        # -------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,                # Transformer内部每个 Token 的特征维度64
            nhead=nhead,                    # 多头注意力机制的头数4，每个头的特征维度为64/4=16
            dim_feedforward=dim_feedforward,# Transformer中前馈神经网络的隐藏层维度128
            dropout=dropout,                # Transformer中dropout的比例0.1
            activation='relu',              # Transformer中前馈神经网络的激活函数ReLU
            batch_first=True                # Transformer中输入输出的batch维度是否在第一维，True表示输入输出的batch维度在第一维
        )

        # -------------------------------------------------
        # Transformer Encoder
        # -------------------------------------------------
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # -------------------------------------------------
        # 输出层，64维 → 1维
        # -------------------------------------------------
        self.fc = nn.Linear(
            d_model,
            1
        )

    def forward(self, x):

        # =================================================
        # Step 1：输入，X:[B, 20, 1]
        # =================================================

        print_shape = False

        if print_shape:
            print("Input:", x.shape)

        # =================================================
        # Step 2：Input Projection,[B, 20, 1] → [B, 20, 64]
        # =================================================

        x = self.input_projection(x)

        if print_shape:
            print("After Projection:", x.shape)

        # =================================================
        # Step 3：Positional Encoding,[B, 20, 64] → [B, 20, 64]
        # =================================================

        x = self.positional_encoding(x)

        if print_shape:
            print("After Positional Encoding:", x.shape)

        # =================================================
        # Step 4：Transformer Encoder,[B, 20, 64] → [B, 20, 64]
        # =================================================

        x = self.transformer(x)

        if print_shape:
            print("After Transformer:", x.shape)

        # =================================================
        # Step 5：取最后一个时间步, [B, 20, 64] → [B, 64]
        # =================================================

        x = x[:, -1, :]   # 利用前面seq_len个时间步的信息来预测下一个时间步的值，作为最终的输出

        if print_shape:
            print("Last Time Step:", x.shape)

        # =================================================
        # Step 6：Linear,[B, 64] → [B, 1]
        # =================================================

        x = self.fc(x)

        if print_shape:
            print("Output:", x.shape)

    
        return x


# =========================================================
# 11. 创建模型
# =========================================================
model = TransformerModel(
    input_size=t_input_size,           # 输入特征维度
    d_model=t_d_model,                 # Transformer模型的隐藏维度
    nhead=t_nhead,                     # 多头注意力机制的头数
    num_layers=t_num_layers,           # Transformer Encoder Layer的层数
    dim_feedforward=t_dim_feedforward, # Transformer中前馈神经网络的隐藏层维度
    dropout=t_dropout                  # Transformer中dropout的比例
)

print("\n模型结构：")
print(model)


# =========================================================
# 12. 损失函数
# =========================================================
criterion = nn.MSELoss()

# =========================================================
# 13. 优化器
# =========================================================
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=t_lr
)


# =========================================================
# 14. 训练
# =========================================================
epochs = t_epochs

loss_list = []

print("\n" + "=" * 60)
print("开始训练")
print("=" * 60)

for epoch in range(epochs):
    
    #训练模型
    model.train()
    
    #梯度清零
    optimizer.zero_grad()

    # 前向传播
    output = model(X_train)

    # 计算 Loss
    loss = criterion(
        output,
        Y_train
    )

    # 反向传播
    loss.backward()

    # 更新参数
    optimizer.step()

    loss_list.append(
        loss.item()
    )
    
    #每20个epoch打印一次loss
    if (epoch + 1) % 20 == 0:

        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Loss: {loss.item():.6f}"
        )


# =========================================================
# 15. 测试
# =========================================================
model.eval()

with torch.no_grad():

    prediction = model(X_test)


# =========================================================
# 16. 恢复原始尺度
# =========================================================
# 先反标准化，得到残差预测
prediction_res = prediction * train_std + train_mean
Y_test_res     = Y_test * train_std + train_mean

# 再加回测试集的趋势
trend_test = a * test_t + b_val

# 注意：Y_test 是滑窗后的目标，对应的时间索引是 test_t[seq_len:]
trend_test_target = trend_test[seq_len:]

prediction_original = prediction_res + trend_test_target.unsqueeze(-1)
Y_test_original    = Y_test_res + trend_test_target.unsqueeze(-1)


# =========================================================
# 17. 计算 MSE、RMSE
# =========================================================
mse = torch.mean(
    (
        prediction_original
        - Y_test_original
    ) ** 2
)

rmse = torch.sqrt(mse)

print("\n==============================")
print("Test Result")
print("==============================")

print(
    "Test Original Scale MSE:",
    mse.item()
)

print(
    "Test Original Scale RMSE:",
    rmse.item()
)


# =========================================================
# 18. 绘制 Loss
# =========================================================
plt.figure(figsize=(10, 5))

plt.plot(
    loss_list
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Transformer Training Loss"
)

plt.grid()

plt.show()


# =========================================================
# 19. 绘制预测结果
# =========================================================
plt.figure(figsize=(12, 5))

plt.plot(
    Y_test_original.numpy(),
    label="True"
)

plt.plot(
    prediction_original.numpy(),
    label="Prediction"
)

plt.xlabel("Time Step")
plt.ylabel("Value")

plt.title(
    "Transformer Time Series Prediction"
)

plt.legend()

plt.grid()

plt.show()


# ============================================================
# ============================================================
# 下面开始进入 Transformer Attention 内部分析
# ============================================================
# ============================================================


# ============================================================
# 21. Attention 分析
# ============================================================

print("\n" + "=" * 60)
print("Transformer Self-Attention Analysis")
print("=" * 60)


# ------------------------------------------------------------
# 取测试集前8个样本
# ------------------------------------------------------------

x_sample = X_test[:8]


print("\n分析样本:")
print(
    "x_sample shape:",
    x_sample.shape
)


# ============================================================
# 22. Input Projection
# ============================================================

model.eval()

with torch.no_grad():

    x = model.input_projection(
        x_sample
    )


print(
    "\nAfter Input Projection:"
)

print(
    "x shape:",
    x.shape
)


# ============================================================
# 23. Positional Encoding
# ============================================================

with torch.no_grad():

    x = model.positional_encoding(
        x
    )


print(
    "\nAfter Positional Encoding:"
)

print(
    "x shape:",
    x.shape
)


# ============================================================
# 24. 取 Transformer 第1层
# ============================================================

encoder_layer = model.transformer.layers[0]

attention = encoder_layer.self_attn


print(
    "\n第1个 Transformer Encoder Layer:"
)

print(
    attention
)


# ============================================================
# 25. 查看 MultiheadAttention 参数
# ============================================================

in_proj_weight = attention.in_proj_weight

in_proj_bias = attention.in_proj_bias


print(
    "\nin_proj_weight shape:",
    in_proj_weight.shape
)

print(
    "in_proj_bias shape:",
    in_proj_bias.shape
)


# ============================================================
# 26. 拆分 WQ、WK、WV
# ============================================================

d_model = t_d_model

W_Q = in_proj_weight[
    :d_model
]

W_K = in_proj_weight[
    d_model:2 * d_model
]

W_V = in_proj_weight[
    2 * d_model:
]


b_Q = in_proj_bias[
    :d_model
]

b_K = in_proj_bias[
    d_model:2 * d_model
]

b_V = in_proj_bias[
    2 * d_model:
]


print(
    "\nQ weight shape:",
    W_Q.shape
)

print(
    "K weight shape:",
    W_K.shape
)

print(
    "V weight shape:",
    W_V.shape
)


# ============================================================
# 27. 计算 Q、K、V
# ============================================================

with torch.no_grad():

    Q = torch.matmul(
        x,
        W_Q.T
    ) + b_Q

    K = torch.matmul(
        x,
        W_K.T
    ) + b_K

    V = torch.matmul(
        x,
        W_V.T
    ) + b_V


print("\n" + "=" * 60)
print("Q / K / V")
print("=" * 60)

print(
    "Q shape:",
    Q.shape
)

print(
    "K shape:",
    K.shape
)

print(
    "V shape:",
    V.shape
)


# ============================================================
# 28. Multi-Head 参数
# ============================================================

batch_size = Q.size(0)

seq_len_actual = Q.size(1)

nhead = t_nhead

head_dim = d_model // nhead


print("\n" + "=" * 60)
print("Multi-Head")
print("=" * 60)

print(
    "d_model:",
    d_model
)

print(
    "nhead:",
    nhead
)

print(
    "head_dim:",
    head_dim
)


# ============================================================
# 29. Q/K/V 分成4个 Head
# ============================================================

Q = Q.view(
    batch_size,
    seq_len_actual,
    nhead,
    head_dim
)

K = K.view(
    batch_size,
    seq_len_actual,
    nhead,
    head_dim
)

V = V.view(
    batch_size,
    seq_len_actual,
    nhead,
    head_dim
)


# ============================================================
# 30. 调整维度
# ============================================================

Q = Q.transpose(
    1,
    2
)

K = K.transpose(
    1,
    2
)

V = V.transpose(
    1,
    2
)


print("\n分成多个 Head 后:")

print(
    "Q shape:",
    Q.shape
)

print(
    "K shape:",
    K.shape
)

print(
    "V shape:",
    V.shape
)


# ============================================================
# 31. 计算 QK^T
# ============================================================

scores = torch.matmul(
    Q,
    K.transpose(
        -2,
        -1
    )
)


print("\n" + "=" * 60)
print("Attention Score")
print("=" * 60)

print(
    "scores shape:",
    scores.shape
)


# ============================================================
# 32. Scaled Dot-Product Attention
# ============================================================

scores = scores / math.sqrt(
    head_dim
)


# ============================================================
# 33. Softmax
# ============================================================

attention_weights = torch.softmax(
    scores,
    dim=-1
)


print("\n" + "=" * 60)
print("Attention Weight")
print("=" * 60)

print(
    "attention_weights shape:",
    attention_weights.shape
)


# ============================================================
# 34. 验证每一行 Attention 是否加起来等于1
# ============================================================

row_sum = attention_weights.sum(
    dim=-1
)


print(
    "\n第一个样本的 Attention 行求和："
)

print(
    row_sum[0]
)


# ============================================================
# 35. 取第一个样本
# ============================================================

sample_attention = attention_weights[0]


print("\n" + "=" * 60)
print("第一个样本")
print("=" * 60)

print(
    "sample_attention shape:",
    sample_attention.shape
)


# ============================================================
# 36. 打印每一个 Head 的 Attention Matrix
# ============================================================

for head in range(nhead):

    print(
        "\n"
        + "-" * 60
    )

    print(
        f"Head {head + 1} Attention Matrix"
    )

    print(
        sample_attention[head]
    )


# ============================================================
# 37. Head 1 热力图
# ============================================================

plt.figure(
    figsize=(8, 7)
)

plt.imshow(
    sample_attention[0].numpy(),
    aspect="auto"
)

plt.colorbar()

plt.xlabel(
    "Key Position"
)

plt.ylabel(
    "Query Position"
)

plt.title(
    "Attention Heatmap - Head 1"
)

plt.xticks(
    range(seq_len_actual)
)

plt.yticks(
    range(seq_len_actual)
)

plt.show()


# ============================================================
# 38. Head 2 热力图
# ============================================================

plt.figure(
    figsize=(8, 7)
)

plt.imshow(
    sample_attention[1].numpy(),
    aspect="auto"
)

plt.colorbar()

plt.xlabel(
    "Key Position"
)

plt.ylabel(
    "Query Position"
)

plt.title(
    "Attention Heatmap - Head 2"
)

plt.xticks(
    range(seq_len_actual)
)

plt.yticks(
    range(seq_len_actual)
)

plt.show()


# ============================================================
# 39. Head 3 热力图
# ============================================================

plt.figure(
    figsize=(8, 7)
)

plt.imshow(
    sample_attention[2].numpy(),
    aspect="auto"
)

plt.colorbar()

plt.xlabel(
    "Key Position"
)

plt.ylabel(
    "Query Position"
)

plt.title(
    "Attention Heatmap - Head 3"
)

plt.xticks(
    range(seq_len_actual)
)

plt.yticks(
    range(seq_len_actual)
)

plt.show()


# ============================================================
# 40. Head 4 热力图
# ============================================================

plt.figure(
    figsize=(8, 7)
)

plt.imshow(
    sample_attention[3].numpy(),
    aspect="auto"
)

plt.colorbar()

plt.xlabel(
    "Key Position"
)

plt.ylabel(
    "Query Position"
)

plt.title(
    "Attention Heatmap - Head 4"
)

plt.xticks(
    range(seq_len_actual)
)

plt.yticks(
    range(seq_len_actual)
)

plt.show()


# ============================================================
# 41. 计算4个 Head 的平均 Attention
# ============================================================

mean_attention = sample_attention.mean(
    dim=0
)


print("\n" + "=" * 60)
print("平均 Attention")
print("=" * 60)

print(
    "mean_attention shape:",
    mean_attention.shape
)


# ============================================================
# 42. 平均 Attention 热力图
# ============================================================

plt.figure(
    figsize=(8, 7)
)

plt.imshow(
    mean_attention.numpy(),
    aspect="auto"
)

plt.colorbar()

plt.xlabel(
    "Key Position"
)

plt.ylabel(
    "Query Position"
)

plt.title(
    "Mean Attention Across Heads"
)

plt.xticks(
    range(seq_len_actual)
)

plt.yticks(
    range(seq_len_actual)
)

plt.show()


# ============================================================
# 43. 找出每个 Query 最关注的位置
# ============================================================

print("\n" + "=" * 60)
print("每个 Query 最关注的位置")
print("=" * 60)


for i in range(seq_len_actual):

    # 第 i 个 Query
    row = mean_attention[i]

    # 找最大 Attention
    max_position = torch.argmax(row).item()

    max_value = row[max_position].item()

    print(
        f"Query位置 {i + 1:2d}"
        f" → 最关注 Key位置 {max_position + 1:2d}"
        f"  Attention = {max_value:.4f}"
    )


# ============================================================
# 44. Attention 总结
# ============================================================

print("\n" + "=" * 60)
print("实验完成")
print("=" * 60)

print("""
Transformer 核心计算：

Q = XW_Q
K = XW_K
V = XW_V

Attention(Q,K,V)
=
Softmax(
    QK^T / sqrt(d_k)
) V

本实验重点观察：

1. Q / K / V 的 Shape
2. Multi-Head 拆分后的 Shape
3. QK^T 的 Shape
4. Attention Weight 的 Shape
5. Attention 热力图
6. 不同 Head 是否关注不同时间位置
""")