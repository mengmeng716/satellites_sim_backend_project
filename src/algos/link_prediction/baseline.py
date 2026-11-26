"""Recursive moving-average baseline for SCFE link-flow predictions."""

import torch
import torch.nn as nn


class MovingAverageBaseline(nn.Module):
    """Use the SCFE tensor interface and average the latest flow samples."""

    def __init__(self, input_len=30, input_dim=11, output_len=30, hidden_dim=512, output_dim=1):
        super().__init__()
        if min(input_len, input_dim, output_len, output_dim) <= 0:
            raise ValueError("Input and output lengths and dimensions must be positive")

        self.output_dim = output_dim
        self.hparams = {
            "input_len": input_len,
            "input_dim": input_dim,
            "output_len": output_len,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
        }

    def forward(self, history_data, future_info=None):
        """Predict normalized link flow with shape (batch, output_len, output_dim).

        The last input feature is normalized flow. Each forecast replaces the
        oldest value in the window before predicting the next step. Extra
        output columns, if requested, repeat the same flow forecast.
        """
        input_len = self.hparams["input_len"]
        input_dim = self.hparams["input_dim"]
        if history_data.ndim != 3 or history_data.shape[1:] != (input_len, input_dim):
            raise ValueError(
                f"Expected history_data shape (batch, {input_len}, {input_dim}), "
                f"got {tuple(history_data.shape)}"
            )

        flow_window = history_data[:, :, -1:]
        predictions = []
        for _ in range(self.hparams["output_len"]):
            next_flow = flow_window.mean(dim=1, keepdim=True)
            predictions.append(next_flow)
            flow_window = torch.cat((flow_window[:, 1:, :], next_flow), dim=1)

        forecast = torch.cat(predictions, dim=1)
        return forecast.expand(-1, -1, self.output_dim).contiguous()
