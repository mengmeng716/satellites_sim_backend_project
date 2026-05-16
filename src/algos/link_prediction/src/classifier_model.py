"""
链路活跃度分类器模型（Link Activity Classifier）

架构设计：
    复用 SCFE 的时序编码思路，但更轻量。
    输入：(batch, 30, 11) 的历史特征序列
    输出：(batch, 1) 的活跃概率（sigmoid后），>0.5 判定为有流量链路

设计原则：
    - 轻量：参数量约为回归器的1/8，推理延迟<20ms
    - 高召回：宁可误报（把零值链路判为有流量），不能漏报（把有流量链路判为零）
    - 可独立训练：不依赖回归器权重
"""

import torch
import torch.nn as nn


class LinkActivityClassifier(nn.Module):
    def __init__(self, input_len=30, input_dim=11, hidden_dim=64):
        """
        参数：
            input_len  : 历史时间步长，与回归器一致，默认30
            input_dim  : 特征维度，与回归器一致，默认11
            hidden_dim : 隐藏层维度，默认64（回归器是512，这里轻量化）
        """
        super().__init__()

        self.hparams = {
            "input_len" : input_len,
            "input_dim" : input_dim,
            "hidden_dim": hidden_dim,
            "model_type": "LinkActivityClassifier",
        }

        # LSTM 提取时序特征
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        # 分类头：取最后一步隐状态 → 二分类
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            # 不加 Sigmoid，使用 BCEWithLogitsLoss 训练更稳定
            # 推理时手动加 sigmoid
        )

    def forward(self, x):
        """
        x: (batch, input_len, input_dim)
        返回: (batch, 1) logits（未经sigmoid）
        """
        out, _ = self.lstm(x)          # (batch, input_len, hidden_dim)
        last   = out[:, -1, :]         # 取最后一步：(batch, hidden_dim)
        logits = self.classifier(last) # (batch, 1)
        return logits

    def predict_proba(self, x):
        """推理时调用，返回 (batch,) 的概率值"""
        logits = self.forward(x)
        return torch.sigmoid(logits).squeeze(-1)