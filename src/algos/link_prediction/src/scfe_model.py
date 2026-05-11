import torch
import torch.nn as nn
import torch.nn.functional as F


class GEU(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.gate = nn.Linear(input_dim, hidden_size)
        self.target = nn.Linear(input_dim, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_gate = F.sigmoid(self.gate(x))
        x_target = self.target(x) * x_gate
        return x_target


class Gate_Concat_Enhance(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.enhance1 = GEU(input_dim, input_dim * 2)
        self.enhance2 = GEU(input_dim * 2, input_dim * 2)
        self.enhance3 = GEU(input_dim * 4, input_dim * 4)
        self.layernorm1 = nn.LayerNorm(input_dim * 2)
        self.layernorm2 = nn.LayerNorm(input_dim * 4)
        self.gelu = nn.GELU()

        self.pro_A = nn.Linear(in_features=input_dim * 9, out_features=output_dim // 2)
        self.pro_B = nn.Linear(in_features=output_dim // 2, out_features=output_dim // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = self.enhance1(x)
        x3 = self.enhance2(self.gelu(self.layernorm1(x2)))
        x4 = torch.cat([x2, x3], dim=2)
        x4 = self.enhance3(self.gelu(self.layernorm2(x4)))
        x = torch.cat([x, x2, x3, x4], dim=2)
        x1 = torch.relu(self.pro_A(x))
        x2 = torch.relu(self.pro_B(x1))
        return torch.cat([x1, x2], dim=2)


class SCFE_without_Future_Clean(nn.Module):
    def __init__(self, input_len=30, input_dim=25, output_len=30, hidden_dim=512):
        super().__init__()
        # 定义链路级输出维度：4条链路 * 2个特征(PredFlow, HeatValue) = 8
        self.output_dim = 8

        self.hparams = {
            "input_len": input_len,
            "input_dim": input_dim,
            "output_len": output_len,
            "hidden_dim": hidden_dim,
            "output_dim": self.output_dim
        }

        # 降维映射层：将 25 维特征映射为 hidden_dim//32
        self.input_proj = nn.Linear(input_dim, hidden_dim // 32)

        # 门控特征拼接增强模块
        self.gate_enhance = Gate_Concat_Enhance(input_dim=hidden_dim // 32, output_dim=hidden_dim)
        self.lstm_encoder = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 4)

        # 扩大输出层容量：现为 output_len * 8 (即 30 * 8 = 240)
        self.history_fc = nn.Linear((hidden_dim // 4) * input_len, output_len * self.output_dim)

    def forward(self, history_data, future_info=None):
        # history_data: (batch, 30, 25)
        x = self.input_proj(history_data)
        x = self.gate_enhance(x)
        out1, _ = self.lstm_encoder(x)

        out1 = self.fc1(out1)
        out1 = out1.flatten(start_dim=1)

        # final_out1 shape: (batch, 240)
        final_out1 = self.history_fc(out1)

        # 解包重塑：将扁平的输出转换为时序链路张量 shape -> (batch, 30, 8)
        final_out1 = final_out1.view(-1, self.hparams["output_len"], self.output_dim)

        return final_out1