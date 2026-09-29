import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. 固定随机种子
# ============================================================

torch.manual_seed(42)
np.random.seed(42)


# ============================================================
# 2. 设置实验参数
# ============================================================

# 输入序列长度：使用过去20个时间点
input_len = 20

# 输出序列长度：预测未来50个时间点
output_len = 50

# 输入变量数量
input_size = 5

# Transformer隐藏维度
d_model = 64

# Multi-Head Attention头数
nhead = 4

# Encoder层数
num_encoder_layers = 2

# Decoder层数
num_decoder_layers = 2

# 前馈神经网络隐藏层维度
dim_feedforward = 128

# Dropout
dropout = 0.1

# 训练轮数
epochs = 300

# 学习率
lr = 0.001

# Batch Size
batch_size = 32


# ============================================================
# 3. 构造多变量时间序列数据
# ============================================================

t = torch.linspace(0, 100, 1000)

x1 = torch.sin(t)
x2 = torch.cos(t)
x3 = torch.sin(2 * t)
x4 = torch.cos(2 * t)
x5 = 0.1 * t


# ============================================================
# 4. 构造目标变量
# ============================================================

y = (
    0.5 * x1
    + 0.3 * x2
    + 0.15 * x3
    + 0.1 * x4
    + 0.1 * x5
)


# ============================================================
# 5. 将5个输入变量组合起来
# ============================================================

X_raw = torch.stack(
    [x1, x2, x3, x4, x5],
    dim=1
)

Y_raw = y.unsqueeze(1)

print("X_raw shape:", X_raw.shape)
print("Y_raw shape:", Y_raw.shape)


# ============================================================
# 6. 划分训练集和测试集
# ============================================================

train_size = 800

X_train_raw = X_raw[:train_size]
Y_train_raw = Y_raw[:train_size]

X_test_raw = X_raw[train_size:]
Y_test_raw = Y_raw[train_size:]

print("\n训练集：")
print("X_train_raw:", X_train_raw.shape)
print("Y_train_raw:", Y_train_raw.shape)

print("\n测试集：")
print("X_test_raw:", X_test_raw.shape)
print("Y_test_raw:", Y_test_raw.shape)


# ============================================================
# 7. 标准化
# ============================================================

X_mean = X_train_raw.mean(dim=0)
X_std = X_train_raw.std(dim=0)

X_train_norm = (
    X_train_raw - X_mean
) / X_std

X_test_norm = (
    X_test_raw - X_mean
) / X_std

Y_mean = Y_train_raw.mean(dim=0)
Y_std = Y_train_raw.std(dim=0)

Y_train_norm = (
    Y_train_raw - Y_mean
) / Y_std

Y_test_norm = (
    Y_test_raw - Y_mean
) / Y_std

print("\nX_mean:")
print(X_mean)

print("\nX_std:")
print(X_std)

print("\nY_mean:")
print(Y_mean)

print("\nY_std:")
print(Y_std)


# ============================================================
# 8. 构造滑动窗口
# ============================================================

def create_sequences(
        X,
        Y,
        input_len,
        output_len
):
    """
    构造：

    过去 input_len 个时间点
        →
    未来 output_len 个时间点
    """

    X_sequences = []
    Y_sequences = []

    total_len = len(X)

    for i in range(
        total_len - input_len - output_len + 1
    ):

        X_seq = X[
            i:i + input_len
        ]

        Y_seq = Y[
            i + input_len:
            i + input_len + output_len
        ]

        X_sequences.append(X_seq)
        Y_sequences.append(Y_seq)

    return (
        torch.stack(X_sequences),
        torch.stack(Y_sequences)
    )


# ============================================================
# 9. 创建训练数据
# ============================================================

X_train, Y_train = create_sequences(
    X_train_norm,
    Y_train_norm,
    input_len,
    output_len
)


# ============================================================
# 10. 创建测试数据
# ============================================================

X_test, Y_test = create_sequences(
    X_test_norm,
    Y_test_norm,
    input_len,
    output_len
)

print("\n滑动窗口之后：")

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)

print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)


# ============================================================
# 11. DataLoader
# ============================================================

train_dataset = torch.utils.data.TensorDataset(
    X_train,
    Y_train
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)


# ============================================================
# 12. Positional Encoding
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_len=500
    ):

        super().__init__()

        position = torch.arange(
            max_len
        ).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2
            ).float()
            * (-np.log(10000.0) / d_model)
        )

        pe = torch.zeros(
            max_len,
            d_model
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe
        )

    def forward(self, x):

        return x + self.pe[
            :, :x.size(1), :
        ]


# ============================================================
# 13. Transformer Encoder-Decoder模型
# ============================================================

class TransformerForecast(
    nn.Module
):

    def __init__(
        self,
        input_size,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        dropout
    ):

        super().__init__()

        # ====================================================
        # 13.1 输入投影
        # ====================================================

        self.input_projection = nn.Linear(
            input_size,
            d_model
        )

        # ====================================================
        # 13.2 Encoder
        # ====================================================

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # ====================================================
        # 13.3 Decoder
        # ====================================================

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )

        # ====================================================
        # 13.4 位置编码
        # ====================================================

        self.pos_encoder = PositionalEncoding(
            d_model
        )

        self.pos_decoder = PositionalEncoding(
            d_model
        )

        # ====================================================
        # 13.5 Decoder输入投影
        # ====================================================

        self.target_projection = nn.Linear(
            1,
            d_model
        )

        # ====================================================
        # 13.6 输出层
        # ====================================================

        self.output_projection = nn.Linear(
            d_model,
            1
        )

    def forward(
        self,
        src,
        tgt
    ):

        src = self.input_projection(src)

        src = self.pos_encoder(src)

        memory = self.encoder(src)

        tgt = self.target_projection(tgt)

        tgt = self.pos_decoder(tgt)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.size(1)
        ).to(tgt.device)

        output = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask
        )

        output = self.output_projection(
            output
        )

        return output


# ============================================================
# 14. 创建模型
# ============================================================

model = TransformerForecast(
    input_size=input_size,
    d_model=d_model,
    nhead=nhead,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    dim_feedforward=dim_feedforward,
    dropout=dropout
)

print("\n模型结构：")
print(model)


# ============================================================
# 15. 损失函数和优化器
# ============================================================

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=lr
)


# ============================================================
# 16. Decoder输入构造
# ============================================================

def create_decoder_input(
    target
):
    """
    Teacher Forcing：

    真实序列：
    y1 y2 y3 y4 y5

    Decoder输入：
    0  y1 y2 y3 y4

    Decoder预测：
    y1 y2 y3 y4 y5
    """

    decoder_input = torch.zeros_like(
        target
    )

    decoder_input[:, 1:, :] = target[
        :, :-1, :
    ]

    return decoder_input


# ============================================================
# 17. 开始训练
# ============================================================

loss_history = []

for epoch in range(epochs):

    model.train()

    total_loss = 0.0

    for X_batch, Y_batch in train_loader:

        decoder_input = create_decoder_input(
            Y_batch
        )

        prediction = model(
            X_batch,
            decoder_input
        )

        loss = criterion(
            prediction,
            Y_batch
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = (
        total_loss
        / len(train_loader)
    )

    loss_history.append(
        average_loss
    )

    if (epoch + 1) % 20 == 0:

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {average_loss:.6f}"
        )


# ============================================================
# 18. 绘制Loss曲线
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    loss_history
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")

plt.title(
    "Transformer Encoder-Decoder Training Loss"
)

plt.grid()

plt.show()


# ============================================================
# 19. 自回归预测
# ============================================================

model.eval()

with torch.no_grad():

    x_sample = X_test[0:1]

    src = model.input_projection(
        x_sample
    )

    src = model.pos_encoder(src)

    memory = model.encoder(src)

    decoder_input = torch.zeros(
        1,
        1,
        1
    )

    predictions = []

    for step in range(output_len):

        tgt = model.target_projection(
            decoder_input
        )

        tgt = model.pos_decoder(tgt)

        tgt_mask = (
            nn.Transformer
            .generate_square_subsequent_mask(
                tgt.size(1)
            )
        )

        output = model.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask
        )

        next_value = model.output_projection(
            output[:, -1:, :]
        )

        predictions.append(
            next_value
        )

        decoder_input = torch.cat(
            [
                decoder_input,
                next_value
            ],
            dim=1
        )

    prediction = torch.cat(
        predictions,
        dim=1
    )


# ============================================================
# 20. 反标准化
# ============================================================

prediction_original = (
    prediction
    * Y_std
    + Y_mean
)

true_original = (
    Y_test[0:1]
    * Y_std
    + Y_mean
)


# ============================================================
# 21. 计算误差
# ============================================================

mse = torch.mean(
    (
        prediction_original
        - true_original
    ) ** 2
)

rmse = torch.sqrt(
    mse
)

mae = torch.mean(
    torch.abs(
        prediction_original
        - true_original
    )
)


print("\n==============================")
print("测试结果")
print("==============================")

print(
    "MSE:",
    mse.item()
)

print(
    "RMSE:",
    rmse.item()
)

print(
    "MAE:",
    mae.item()
)


# ============================================================
# 22. 绘制预测结果
# ============================================================

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    true_original.squeeze().numpy(),
    label="True"
)

plt.plot(
    prediction_original.squeeze().numpy(),
    label="Prediction"
)

plt.xlabel(
    "Future Time Step"
)

plt.ylabel(
    "Target Value"
)

plt.title(
    "20 → 50 Multivariate Transformer Forecast"
)

plt.legend()

plt.grid()

plt.show()


# ============================================================
# 23. Attention热力图
#
# 重要说明：
#
# 这里没有修改Transformer的结构、参数、训练方式。
#
# 前面的1~22部分就是原来的实验。
#
# Attention只在训练完成之后重新计算，
# 因此不会参与训练，也不会改变MSE/RMSE/MAE。
# ============================================================


def get_attention_weights(
    attention_module,
    query,
    key,
    value,
    attn_mask=None
):
    """
    使用已经训练好的MultiheadAttention，
    重新计算Attention权重。

    不修改模型参数。
    """

    try:

        _, weights = attention_module(
            query,
            key,
            value,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=False
        )

    except TypeError:

        _, weights = attention_module(
            query,
            key,
            value,
            attn_mask=attn_mask,
            need_weights=True
        )

    return weights


# ============================================================
# 24. Encoder Attention
# ============================================================

with torch.no_grad():

    encoder_input = model.input_projection(
        x_sample
    )

    encoder_input = model.pos_encoder(
        encoder_input
    )

    encoder_output = encoder_input

    encoder_attention_list = []

    for layer in model.encoder.layers:

        attn = get_attention_weights(
            layer.self_attn,
            encoder_output,
            encoder_output,
            encoder_output
        )

        encoder_attention_list.append(
            attn
        )

        encoder_output = layer(
            encoder_output
        )


# ============================================================
# 25. 取最后一层Encoder Attention
# ============================================================

encoder_attention = (
    encoder_attention_list[-1]
)

print("\n==============================")
print("Encoder Attention")
print("==============================")

print(
    "Encoder Attention Shape:",
    encoder_attention.shape
)


if encoder_attention.dim() == 4:

    encoder_attention_mean = (
        encoder_attention[0]
        .mean(dim=0)
        .cpu()
        .numpy()
    )

else:

    encoder_attention_mean = (
        encoder_attention[0]
        .cpu()
        .numpy()
    )


# ============================================================
# 26. Encoder Self-Attention热力图
# ============================================================

plt.figure(
    figsize=(8, 7)
)

plt.imshow(
    encoder_attention_mean,
    aspect="auto"
)

plt.colorbar()

plt.xlabel(
    "Historical Key Time Step"
)

plt.ylabel(
    "Historical Query Time Step"
)

plt.title(
    "Encoder Self-Attention"
)

plt.xticks(
    range(input_len)
)

plt.yticks(
    range(input_len)
)

plt.show()


# ============================================================
# 27. Decoder Attention
# ============================================================
#
# 自回归预测结束后：
#
# decoder_input长度 = 51
#
# 第一个值是初始0，
# 后面50个是预测值。
#
# 最后一次真正用于预测y50的Decoder输入：
#
# 0, y_hat1, ..., y_hat49
#
# 所以去掉最后一个y_hat50。
# ============================================================

decoder_attention_input = (
    decoder_input[:, :-1, :]
)


with torch.no_grad():

    tgt = model.target_projection(
        decoder_attention_input
    )

    tgt = model.pos_decoder(
        tgt
    )

    tgt_mask = (
        nn.Transformer
        .generate_square_subsequent_mask(
            tgt.size(1)
        )
    )

    decoder_output = tgt

    decoder_self_attention_list = []

    decoder_cross_attention_list = []

    for layer in model.decoder.layers:

        # ====================================================
        # Decoder Self-Attention
        # ====================================================

        self_attn = get_attention_weights(
            layer.self_attn,
            decoder_output,
            decoder_output,
            decoder_output,
            attn_mask=tgt_mask
        )

        decoder_self_attention_list.append(
            self_attn
        )

        # ----------------------------------------------------
        # 重新执行这一层的Self-Attention
        # ----------------------------------------------------

        self_output = layer.self_attn(
            decoder_output,
            decoder_output,
            decoder_output,
            attn_mask=tgt_mask,
            need_weights=False
        )[0]

        decoder_output_after_self = (
            layer.norm1(
                decoder_output
                + layer.dropout1(
                    self_output
                )
            )
        )

        # ====================================================
        # Decoder Cross-Attention
        # ====================================================

        cross_attn = get_attention_weights(
            layer.multihead_attn,
            decoder_output_after_self,
            encoder_output,
            encoder_output
        )

        decoder_cross_attention_list.append(
            cross_attn
        )

        # ----------------------------------------------------
        # 继续正常执行这一层
        # ----------------------------------------------------

        cross_output = layer.multihead_attn(
            decoder_output_after_self,
            encoder_output,
            encoder_output,
            need_weights=False
        )[0]

        decoder_output_after_cross = (
            layer.norm2(
                decoder_output_after_self
                + layer.dropout2(
                    cross_output
                )
            )
        )

        # ====================================================
        # Feed Forward
        # ====================================================

        ff = layer.linear2(
            layer.dropout(
                layer.activation(
                    layer.linear1(
                        decoder_output_after_cross
                    )
                )
            )
        )

        decoder_output = (
            layer.norm3(
                decoder_output_after_cross
                + layer.dropout3(ff)
            )
        )


# ============================================================
# 28. Decoder Self-Attention热力图
# ============================================================

decoder_self_attention = (
    decoder_self_attention_list[-1]
)

print("\n==============================")
print("Decoder Self-Attention")
print("==============================")

print(
    "Shape:",
    decoder_self_attention.shape
)

if decoder_self_attention.dim() == 4:

    decoder_self_attention_mean = (
        decoder_self_attention[0]
        .mean(dim=0)
        .cpu()
        .numpy()
    )

else:

    decoder_self_attention_mean = (
        decoder_self_attention[0]
        .cpu()
        .numpy()
    )


plt.figure(
    figsize=(10, 8)
)

plt.imshow(
    decoder_self_attention_mean,
    aspect="auto"
)

plt.colorbar()

plt.xlabel(
    "Decoder Key Time Step"
)

plt.ylabel(
    "Decoder Query Time Step"
)

plt.title(
    "Decoder Self-Attention"
)

plt.show()


# ============================================================
# 29. Decoder Cross-Attention热力图
#
# 50个未来时间点
#
# ↓
#
# 20个历史时间点
# ============================================================

decoder_cross_attention = (
    decoder_cross_attention_list[-1]
)

print("\n==============================")
print("Decoder Cross-Attention")
print("==============================")

print(
    "Shape:",
    decoder_cross_attention.shape
)

if decoder_cross_attention.dim() == 4:

    decoder_cross_attention_mean = (
        decoder_cross_attention[0]
        .mean(dim=0)
        .cpu()
        .numpy()
    )

else:

    decoder_cross_attention_mean = (
        decoder_cross_attention[0]
        .cpu()
        .numpy()
    )


plt.figure(
    figsize=(10, 8)
)

plt.imshow(
    decoder_cross_attention_mean,
    aspect="auto"
)

plt.colorbar()

plt.xlabel(
    "Encoder Historical Time Step"
)

plt.ylabel(
    "Decoder Future Time Step"
)

plt.title(
    "Decoder Cross-Attention"
)

plt.xticks(
    range(input_len)
)

plt.yticks(
    range(0, output_len, 5)
)

plt.show()


# ============================================================
# 30. 观察Future Step 1 / 10 / 25 / 50
# ============================================================

selected_steps = [
    0,
    9,
    24,
    49
]

for step in selected_steps:

    attention = (
        decoder_cross_attention_mean[
            step
        ]
    )

    plt.figure(
        figsize=(10, 4)
    )

    plt.bar(
        range(1, input_len + 1),
        attention
    )

    plt.xlabel(
        "Encoder Historical Time Step"
    )

    plt.ylabel(
        "Attention"
    )

    plt.title(
        f"Cross-Attention "
        f"at Future Step {step + 1}"
    )

    plt.xticks(
        range(1, input_len + 1)
    )

    plt.grid()

    plt.show()


print("\n==============================")
print("Attention可视化完成")
print("==============================")
