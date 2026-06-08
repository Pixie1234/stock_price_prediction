import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data_pipeline import (
    N_FEATURES,
    OPEN_IDX,
    CLOSE_IDX,
    BODY_PCT_IDX,
    HIGH_IDX,
    LOW_IDX,
    VOLUME_IDX,
    SEQ_LEN,
    DIRECT_HORIZON,
    inverse_transform_col,
)


DEVICE = torch.device("cpu")


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:seq_len].unsqueeze(0)


class InformerStyleEncoder(nn.Module):
    """A practical Informer-like encoder for small numeric time series.

    This is not a full ProbSparse Informer implementation, but it follows the
    same idea: attention-based temporal modeling with an encoder stack.
    """

    def __init__(
        self,
        n_features: int = N_FEATURES,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 64,
        dropout: float = 0.15,
        close_loss_weight: float = 2.0,
        out_dim: int = 2,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.close_loss_weight = float(close_loss_weight)

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)

        # Next-step prediction from last timestep.
        last = h[:, -1, :]
        return self.head(last)

    def weighted_smooth_l1(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        # y_true/y_pred: (batch, 2) in scaled log-return space.
        # SmoothL1 per element, weighted on Close.
        loss_open = nn.functional.smooth_l1_loss(
            y_pred[:, 0], y_true[:, 0], beta=1.0, reduction="mean"
        )
        loss_close = nn.functional.smooth_l1_loss(
            y_pred[:, 1], y_true[:, 1], beta=1.0, reduction="mean"
        )
        return loss_open + self.close_loss_weight * loss_close


@dataclass
class InformerConfig:
    # Keep the model small so training is fast on CPU.
    d_model: int = 24
    n_heads: int = 4
    n_layers: int = 2
    dim_feedforward: int = 48
    dropout: float = 0.15
    close_loss_weight: float = 2.0

    lr: float = 1e-3
    batch_size: int = 128
    max_epochs: int = 25
    patience: int = 8


def _make_model(cfg: InformerConfig) -> InformerStyleEncoder:
    return InformerStyleEncoder(
        n_features=N_FEATURES,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        close_loss_weight=cfg.close_loss_weight,
    ).to(DEVICE)


def _make_direct_model(cfg: InformerConfig, horizon: int) -> InformerStyleEncoder:
    return InformerStyleEncoder(
        n_features=N_FEATURES,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        close_loss_weight=cfg.close_loss_weight,
        out_dim=horizon * 2,
    ).to(DEVICE)


def predict_informer(model: nn.Module, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    model.eval()
    preds = []
    dl = DataLoader(torch.from_numpy(X).float(), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for xb in dl:
            xb = xb.to(DEVICE)
            yb = model(xb)
            preds.append(yb.cpu().numpy())
    return np.concatenate(preds, axis=0)


def train_informer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_path: str,
    cfg: InformerConfig,
) -> nn.Module:
    os.makedirs(os.path.dirname(model_path) if model_path and os.path.dirname(model_path) else ".", exist_ok=True)

    model = _make_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    ds_train = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    ds_val = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False)

    best_val = float("inf")
    best_state = None
    patience_left = cfg.patience

    for epoch in range(cfg.max_epochs):
        model.train()
        for xb, yb in dl_train:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = model.weighted_smooth_l1(yb, pred)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # Val
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in dl_val:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                pred = model(xb)
                val_losses.append(float(model.weighted_smooth_l1(yb, pred).cpu().item()))
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__}, model_path)
    return model


def load_or_train_informer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_path: str,
    cfg: Optional[InformerConfig] = None,
):
    cfg = cfg or InformerConfig()

    if model_path and os.path.exists(model_path):
        try:
            blob = torch.load(model_path, map_location=DEVICE)
            model = _make_model(cfg)
            model.load_state_dict(blob["state_dict"], strict=True)
            model.eval()
            return model, True
        except Exception:
            # Fall back to training.
            pass

    model = train_informer(X_train, y_train, X_val, y_val, model_path, cfg)
    return model, False


def train_direct_informer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_path: str,
    horizon: int = DIRECT_HORIZON,
    cfg: Optional[InformerConfig] = None,
) -> nn.Module:
    cfg = cfg or InformerConfig()
    os.makedirs(os.path.dirname(model_path) if model_path and os.path.dirname(model_path) else ".", exist_ok=True)

    model = _make_direct_model(cfg, horizon)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    ds_train = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    ds_val = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False)

    best_val = float("inf")
    best_state = None
    patience_left = cfg.patience

    for epoch in range(cfg.max_epochs):
        model.train()
        for xb, yb in dl_train:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = nn.functional.smooth_l1_loss(pred, yb, reduction="mean")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in dl_val:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                pred = model(xb)
                val_losses.append(float(nn.functional.smooth_l1_loss(pred, yb, reduction="mean").cpu().item()))
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__, "horizon": horizon}, model_path)
    return model


def load_or_train_direct_informer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_path: str,
    horizon: int = DIRECT_HORIZON,
    cfg: Optional[InformerConfig] = None,
):
    cfg = cfg or InformerConfig()

    if model_path and os.path.exists(model_path):
        try:
            blob = torch.load(model_path, map_location=DEVICE)
            model = _make_direct_model(cfg, horizon)
            model.load_state_dict(blob["state_dict"], strict=True)
            model.eval()
            return model, True
        except Exception:
            pass

    model = train_direct_informer(X_train, y_train, X_val, y_val, model_path, horizon=horizon, cfg=cfg)
    return model, False


def estimate_direct_bias_informer(model: nn.Module, X_val: np.ndarray, y_val: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Estimate validation-set mean residuals in scaled output space."""
    if X_val is None or y_val is None or len(X_val) == 0:
        out_dim = int(getattr(model.head[-1], "out_features", 2))
        return np.zeros(out_dim, dtype=float)
    preds = predict_informer(model, X_val, batch_size=batch_size)
    return np.mean(preds - y_val, axis=0)


def forecast_ohlcv_informer(
    model: nn.Module,
    last_sequence: np.ndarray,
    days: int,
    scaler,
    raw_ohlcv: np.ndarray,
    batch_eval: int = 256,
) -> Dict[str, list]:
    """Autoregressive multi-step forecast for Open/Close (mirrors LSTM logic)."""
    model.eval()

    predictions = {
        "open_prices": [],
        "close_prices": [],
        "open_returns": [],
        "close_returns": [],
    }

    seq = last_sequence.copy()
    base_prices = raw_ohlcv[-1]
    current_price = {"Open": float(base_prices[0]), "Close": float(base_prices[3])}

    # Prevent open-feedback drift: blend predicted Open return with the last
    # Open return already present in the input sequence window.
    OPEN_FEEDBACK_DAMPING = 0.75
    OPEN_MAGNITUDE_DAMPING = 0.6

    # Stabilize roll-forward for High/Low/Volume features during multi-step forecast.
    # We approximate future OHLC shape using recent averages plus the magnitude
    # of predicted price movement.
    recent_window = seq[-20:].copy()
    mean_high_scaled = float(np.mean(recent_window[:, HIGH_IDX]))
    mean_low_scaled = float(np.mean(recent_window[:, LOW_IDX]))
    mean_volume_scaled = float(np.mean(recent_window[:, VOLUME_IDX]))

    for _ in range(days):

        x_in = torch.from_numpy(seq.reshape(1, SEQ_LEN, N_FEATURES)).float().to(DEVICE)
        with torch.no_grad():
            pred_scaled = model(x_in).cpu().numpy()[0]

        open_ret_scaled = float(pred_scaled[0])
        body_pct_scaled = float(pred_scaled[1])

        prev_open_ret_scaled = float(seq[-1, OPEN_IDX])
        open_ret_scaled = (
            prev_open_ret_scaled * (1.0 - OPEN_FEEDBACK_DAMPING)
            + open_ret_scaled * OPEN_FEEDBACK_DAMPING
        )

        open_ret_scaled = open_ret_scaled * OPEN_MAGNITUDE_DAMPING

        open_ret = float(inverse_transform_col(open_ret_scaled, OPEN_IDX, scaler))
        body_pct = float(inverse_transform_col(body_pct_scaled, BODY_PCT_IDX, scaler))
        body_pct = max(body_pct, -0.999)

        prev_open_price = current_price["Open"]
        prev_close_price = current_price["Close"]
        current_price["Open"] = float(prev_open_price * np.exp(open_ret))
        current_price["Close"] = float(current_price["Open"] * (1.0 + body_pct))

        close_ret = float(np.log((current_price["Close"] + 1e-12) / (prev_close_price + 1e-12)))
        mean_open = float(scaler.mean_[OPEN_IDX])
        std_open = float(scaler.scale_[OPEN_IDX])
        open_ret_scaled = (open_ret - mean_open) / (std_open + 1e-12)
        mean_body = float(scaler.mean_[BODY_PCT_IDX])
        std_body = float(scaler.scale_[BODY_PCT_IDX])
        body_pct_scaled = (body_pct - mean_body) / (std_body + 1e-12)
        mean_close = float(scaler.mean_[CLOSE_IDX])
        std_close = float(scaler.scale_[CLOSE_IDX])
        close_ret_scaled = (close_ret - mean_close) / (std_close + 1e-12)

        predictions["open_returns"].append(open_ret)
        predictions["close_returns"].append(close_ret)
        predictions["open_prices"].append(round(current_price["Open"], 2))
        predictions["close_prices"].append(round(current_price["Close"], 2))

        new_seq = np.roll(seq, -1, axis=0)
        next_feat = new_seq[-1].copy()

        next_feat[OPEN_IDX] = open_ret_scaled
        next_feat[CLOSE_IDX] = close_ret_scaled
        next_feat[BODY_PCT_IDX] = body_pct_scaled

        # Heuristic: approximate High/Low/Volume features using recent averages
        # and predicted price movement magnitude.
        abs_ret = abs(close_ret_scaled)
        next_feat[HIGH_IDX] = abs_ret * 0.8 + mean_high_scaled * 0.2
        next_feat[LOW_IDX] = -abs_ret * 0.8 + mean_low_scaled * 0.2
        next_feat[VOLUME_IDX] = mean_volume_scaled

        new_seq[-1] = next_feat
        seq = new_seq

    return predictions


def forecast_direct_ohlcv_informer(
    model: nn.Module,
    last_sequence: np.ndarray,
    horizon: int,
    scaler,
    raw_ohlcv: np.ndarray,
    bias: Optional[np.ndarray] = None,
) -> Dict[str, list]:
    """Direct multi-horizon forecast without recursive feature roll-forward."""
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(last_sequence.reshape(1, SEQ_LEN, N_FEATURES)).float().to(DEVICE)).cpu().numpy()[0]
    pred_scaled = np.asarray(pred_scaled)
    total_horizon = int(pred_scaled.shape[0] // 2)
    pred_scaled = pred_scaled.reshape(total_horizon, 2)
    horizon = min(int(horizon), total_horizon)

    if bias is not None:
        bias = np.asarray(bias).reshape(total_horizon, 2)
        pred_scaled = pred_scaled - bias[:total_horizon]

    base_prices = raw_ohlcv[-1]
    current_open = float(base_prices[0])
    current_close = float(base_prices[3])

    predictions = {
        "open_prices": [],
        "close_prices": [],
        "open_returns": [],
        "close_returns": [],
    }

    for i in range(horizon):
        open_ret = float(inverse_transform_col(pred_scaled[i, 0], OPEN_IDX, scaler))
        body_pct = float(inverse_transform_col(pred_scaled[i, 1], BODY_PCT_IDX, scaler))
        body_pct = max(body_pct, -0.999)
        prev_close = current_close
        current_open *= float(np.exp(open_ret))
        current_close = float(current_open * (1.0 + body_pct))
        close_ret = float(np.log((current_close + 1e-12) / (prev_close + 1e-12)))
        predictions["open_returns"].append(open_ret)
        predictions["close_returns"].append(close_ret)
        predictions["open_prices"].append(round(current_open, 2))
        predictions["close_prices"].append(round(current_close, 2))

    return predictions
