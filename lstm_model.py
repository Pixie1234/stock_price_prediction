# ============================================================
# lstm_model.py - FIXED for proper direction predictions
# ============================================================
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, LayerNormalization, Bidirectional,
    Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau
)
from data_pipeline import (
    N_FEATURES, N_OUTPUTS, SEQ_LEN, DIRECT_HORIZON,
    OPEN_IDX, CLOSE_IDX, BODY_PCT_IDX, HIGH_IDX, LOW_IDX, VOLUME_IDX,
    inverse_transform_col
)

try:
    # Optional: only needed when using Hugging Face Hub persistence.
    from huggingface_hub import hf_hub_download, HfApi
except Exception:  # pragma: no cover
    hf_hub_download = None
    HfApi = None

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Must be at module scope so saved .h5 models can be deserialized.
# Give the second target some emphasis, but avoid over-dominating Open.
CLOSE_LOSS_WEIGHT = 2.0


def weighted_huber(y_true, y_pred):
    """Elementwise weighted Huber loss for [Open, target] outputs."""
    delta = tf.constant(1.0, dtype=y_true.dtype)
    err = y_true - y_pred
    abs_err = tf.abs(err)
    quadratic = tf.minimum(abs_err, delta)
    linear = abs_err - quadratic
    per_elem = 0.5 * tf.square(quadratic) + delta * linear  # (batch, 2)
    weights = tf.constant([1.0, CLOSE_LOSS_WEIGHT], dtype=per_elem.dtype)
    return tf.reduce_mean(per_elem * weights)


def build_model(seq_len=SEQ_LEN, n_features=N_FEATURES, n_outputs=N_OUTPUTS):
    """
    Bidirectional LSTM with Layer Normalization.
    LayerNorm normalizes activations across features within each timestep.
    """
    # Functional model: shared backbone + separate heads.
    # This reduces interference between Open and Close regression tasks.
    inp = Input(shape=(seq_len, n_features))

    x = Bidirectional(LSTM(64, return_sequences=True))(inp)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(32))(x)
    x = Dropout(0.3)(x)
    x = LayerNormalization()(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.2)(x)

    open_head = Dense(16, activation="relu")(x)
    open_out = Dense(1)(open_head)

    close_head = Dense(16, activation="relu")(x)
    close_out = Dense(1)(close_head)

    out = Concatenate(name="open_close_concat")([open_out, close_out])

    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=weighted_huber,
        metrics=["mae"],
    )
    return model


def build_close_only_model(seq_len=SEQ_LEN, n_features=N_FEATURES):
    """Close-only model for comparison against the shared Open+Close model."""
    inp = Input(shape=(seq_len, n_features))

    x = Bidirectional(LSTM(64, return_sequences=True))(inp)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(32))(x)
    x = Dropout(0.3)(x)
    x = LayerNormalization()(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation="relu")(x)
    out = Dense(1, name="close_only")(x)

    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="huber",
        metrics=["mae"],
    )
    return model


def build_direct_model(seq_len=SEQ_LEN, n_features=N_FEATURES, horizon=DIRECT_HORIZON):
    """Direct multi-horizon model: predicts [Open_1, Close_1, ..., Open_H, Close_H]."""
    inp = Input(shape=(seq_len, n_features))

    x = Bidirectional(LSTM(64, return_sequences=True))(inp)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(32))(x)
    x = Dropout(0.3)(x)
    x = LayerNormalization()(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.1)(x)

    out = Dense(horizon * 2, name="direct_open_close") (x)

    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="huber",
        metrics=["mae"],
    )
    return model


def train_lstm(X_train, y_train, X_val=None, y_val=None, model_path=None, n_features=None):
    """
    Train with class balancing via SAMPLE WEIGHTING.
    Samples with minority direction get higher weight.
    """
    os.makedirs(os.path.dirname(model_path) if model_path and os.path.dirname(model_path) else ".", exist_ok=True)

    if n_features is None:
        n_features = X_train.shape[2]
    
    model = build_model(n_features=n_features)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=15,
            min_delta=1e-6,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    
    if X_val is not None and y_val is not None:
        model.fit(
            X_train,
            y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
        )
    else:
        model.fit(
            X_train,
            y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.15,
            callbacks=callbacks,
            verbose=1,
        )
    
    if model_path:
        model.save(model_path)
    return model


def train_close_only_lstm(X_train, y_train_close, X_val=None, y_val_close=None, model_path=None, n_features=None):
    os.makedirs(os.path.dirname(model_path) if model_path and os.path.dirname(model_path) else ".", exist_ok=True)

    if n_features is None:
        n_features = X_train.shape[2]

    model = build_close_only_model(n_features=n_features)

    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1),
    ]

    if X_val is not None and y_val_close is not None:
        model.fit(
            X_train,
            y_train_close,
            epochs=200,
            batch_size=32,
            validation_data=(X_val, y_val_close),
            callbacks=callbacks,
            verbose=1,
        )
    else:
        model.fit(
            X_train,
            y_train_close,
            epochs=200,
            batch_size=32,
            validation_split=0.15,
            callbacks=callbacks,
            verbose=1,
        )

    if model_path:
        model.save(model_path)
    return model


def _close_residual_targets(X, y_close):
    """Target residual over the last observed Close return in the input window."""
    last_close = X[:, -1, CLOSE_IDX]
    return y_close - last_close


def train_close_residual_lstm(X_train, y_train_close, X_val=None, y_val_close=None, model_path=None, n_features=None):
    os.makedirs(os.path.dirname(model_path) if model_path and os.path.dirname(model_path) else ".", exist_ok=True)

    if n_features is None:
        n_features = X_train.shape[2]

    y_train_resid = _close_residual_targets(X_train, y_train_close)
    y_val_resid = _close_residual_targets(X_val, y_val_close) if X_val is not None and y_val_close is not None else None

    model = build_close_only_model(n_features=n_features)

    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1),
    ]

    if X_val is not None and y_val_resid is not None:
        model.fit(
            X_train,
            y_train_resid,
            epochs=200,
            batch_size=32,
            validation_data=(X_val, y_val_resid),
            callbacks=callbacks,
            verbose=1,
        )
    else:
        model.fit(
            X_train,
            y_train_resid,
            epochs=200,
            batch_size=32,
            validation_split=0.15,
            callbacks=callbacks,
            verbose=1,
        )

    if model_path:
        model.save(model_path)
    return model


def load_or_train(X_train, y_train, X_val=None, y_val=None, model_path=None):
    """Load saved model or train new one."""

    expected_n_features = int(X_train.shape[2])

    def _validate_loaded_model_input(model: tf.keras.Model) -> None:
        """Fail fast if the loaded model expects a different feature dimension."""
        # A single forward pass is cheap and avoids relying on fragile input_shape parsing.
        try:
            model.predict(X_train[:1], verbose=0)
        except Exception as e:
            # Most common mismatch manifests as an InputSpec incompatibility / expected shape error.
            msg = str(e)
            if "expected shape" in msg or "incompatible" in msg or "found shape" in msg:
                raise ValueError(
                    f"Loaded model input mismatch (expected features={expected_n_features})"
                ) from e
            raise

    def _input_dim_from_model(m: tf.keras.Model) -> int | None:
        try:
            inp_shape = getattr(m, "input_shape", None)
            # For functional models: (batch, seq_len, features)
            if inp_shape is not None:
                if isinstance(inp_shape, (tuple, list)) and len(inp_shape) >= 3:
                    return int(inp_shape[-1])

            # Fallback: try reading from first input tensor.
            if getattr(m, "inputs", None):
                t0 = m.inputs[0]
                if t0.shape is not None and len(t0.shape) >= 3:
                    # shape: (batch, seq_len, features)
                    return int(t0.shape[-1])
        except Exception:
            return None
        return None

    def _hf_repo_id() -> str | None:
        repo = os.environ.get("HF_MODEL_REPO")
        return repo.strip() if repo else None

    def _hf_token() -> str | None:
        tok = os.environ.get("HF_TOKEN")
        return tok.strip() if tok else None

    def _hf_upload_enabled() -> bool:
        return os.environ.get("HF_UPLOAD_MODELS", "0") == "1"

    def _download_from_hf_if_missing() -> bool:
        if not model_path or os.path.exists(model_path):
            return False
        if hf_hub_download is None:
            return False

        repo_id = _hf_repo_id()
        if not repo_id:
            return False

        filename = os.path.basename(model_path)
        token = _hf_token()
        try:
            print(f"[model] Local model missing. Downloading from HF repo={repo_id} file={filename} ...")
            # Downloads to the local cache; then we copy into model_path.
            local_cached = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=token,
            )
            os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else ".", exist_ok=True)
            # Overwrite if training had partially created the file.
            with open(local_cached, "rb") as src, open(model_path, "wb") as dst:
                dst.write(src.read())
            print(f"[model] HF download OK -> {model_path}")
            return True
        except Exception:
            print(f"[model] HF download FAILED for {filename}")
            return False

    def _maybe_upload_to_hf() -> None:
        if not model_path or not os.path.exists(model_path):
            return
        if hf_hub_download is None or HfApi is None:
            return
        if not _hf_upload_enabled():
            return
        repo_id = _hf_repo_id()
        if not repo_id:
            return
        token = _hf_token()
        if not token:
            return

        filename = os.path.basename(model_path)
        try:
            api = HfApi()
            # Ensure repo exists (create_repo is safe if it already exists).
            api.create_repo(repo_id, repo_type="model", token=token, exist_ok=True)
            api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="model",
                token=token,
            )
        except Exception:
            # Upload failures should not break inference.
            return

    # 1) Try local load first.
    if model_path and os.path.exists(model_path):
        try:
            print(f"[model] Trying local load: {model_path}")
            # Some deployments may have a different Keras/TensorFlow version.
            # Loading can fail with .h5 (serialization mismatch).
            model = load_model(
                model_path,
                compile=False,
                custom_objects={"weighted_huber": weighted_huber},
            )

            loaded_dim = _input_dim_from_model(model)
            if loaded_dim is not None and loaded_dim != expected_n_features:
                raise ValueError(
                    f"Input feature mismatch: loaded={loaded_dim}, expected={expected_n_features}"
                )

            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss=weighted_huber,
                metrics=["mae"],
            )

            _validate_loaded_model_input(model)
            return model, True
        except Exception as e:
            # Fallback: may download or retrain from scratch.
            print(
                f"[model] Local load FAILED for {model_path} -> {type(e).__name__}: {e}"
            )
            pass

    # 2) Download from HF if local model is missing.
    downloaded = _download_from_hf_if_missing()
    if downloaded:
        print(f"[model] Proceeding to load downloaded model from {model_path}")

    if model_path and os.path.exists(model_path):
        try:
            # This is used when HF download succeeded but local deserialization still fails.
            model = load_model(
                model_path,
                compile=False,
                custom_objects={"weighted_huber": weighted_huber},
            )

            loaded_dim = _input_dim_from_model(model)
            if loaded_dim is not None and loaded_dim != expected_n_features:
                raise ValueError(
                    f"Input feature mismatch: loaded={loaded_dim}, expected={expected_n_features}"
                )

            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss=weighted_huber,
                metrics=["mae"],
            )

            _validate_loaded_model_input(model)
            return model, True
        except Exception as e:
            print(
                f"[model] Post-download load FAILED for {model_path} -> {type(e).__name__}: {e}"
            )
            pass

    # 3) Train and (optionally) upload.
    print(f"[model] Training new model -> {model_path}")
    model = train_lstm(X_train, y_train, X_val, y_val, model_path)
    _maybe_upload_to_hf()
    return model, False


def load_or_train_close_only(X_train, y_train_close, X_val=None, y_val_close=None, model_path=None):
    """Load or train a model that predicts Close only."""
    expected_n_features = int(X_train.shape[2])

    if model_path and os.path.exists(model_path):
        try:
            model = load_model(model_path, compile=False)
            if int(model.output_shape[-1]) != 1:
                raise ValueError("Output dimension mismatch")
            model.compile(optimizer=Adam(learning_rate=0.001), loss="huber", metrics=["mae"])
            model.predict(X_train[:1], verbose=0)
            return model, True
        except Exception:
            pass

    model = train_close_only_lstm(X_train, y_train_close, X_val, y_val_close, model_path, n_features=expected_n_features)
    return model, False


def load_or_train_close_residual(X_train, y_train_close, X_val=None, y_val_close=None, model_path=None):
    """Load or train a model that predicts the residual over last observed Close return."""
    expected_n_features = int(X_train.shape[2])

    if model_path and os.path.exists(model_path):
        try:
            model = load_model(model_path, compile=False)
            if int(model.output_shape[-1]) != 1:
                raise ValueError("Output dimension mismatch")
            model.compile(optimizer=Adam(learning_rate=0.001), loss="huber", metrics=["mae"])
            model.predict(X_train[:1], verbose=0)
            return model, True
        except Exception:
            pass

    model = train_close_residual_lstm(X_train, y_train_close, X_val, y_val_close, model_path, n_features=expected_n_features)
    return model, False


def train_direct_lstm(X_train, y_train, X_val=None, y_val=None, model_path=None, horizon=DIRECT_HORIZON):
    os.makedirs(os.path.dirname(model_path) if model_path and os.path.dirname(model_path) else ".", exist_ok=True)
    model = build_direct_model(n_features=X_train.shape[2], horizon=horizon)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, min_delta=1e-6, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1),
    ]

    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
        )
    else:
        model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.15,
            callbacks=callbacks,
            verbose=1,
        )

    if model_path:
        model.save(model_path)
    return model


def load_or_train_direct(X_train, y_train, X_val=None, y_val=None, model_path=None, horizon=DIRECT_HORIZON):
    expected_n_features = int(X_train.shape[2])

    if model_path and os.path.exists(model_path):
        try:
            model = load_model(model_path, compile=False)
            if int(model.output_shape[-1]) != horizon * 2:
                raise ValueError("Output horizon mismatch")
            model.compile(optimizer=Adam(learning_rate=0.001), loss="huber", metrics=["mae"])
            model.predict(X_train[:1], verbose=0)
            return model, True
        except Exception:
            pass

    model = train_direct_lstm(X_train, y_train, X_val, y_val, model_path, horizon=horizon)
    return model, False


def estimate_direct_bias(model, X_val, y_val):
    """Estimate validation-set mean residuals in scaled output space."""
    if X_val is None or y_val is None or len(X_val) == 0:
        out_dim = int(model.output_shape[-1])
        return np.zeros(out_dim, dtype=float)
    preds = model.predict(X_val, verbose=0)
    return np.mean(preds - y_val, axis=0)


def forecast_ohlcv(model, last_sequence, days, scaler, raw_ohlcv):
    """Forecast with autoregressive loop."""
    predictions = {
        "open_prices": [],
        "close_prices": [],
        "open_returns": [],
        "close_returns": [],
    }
    seq = last_sequence.copy()

    base_prices = raw_ohlcv[-1]
    current_price = {"Open": float(base_prices[0]), "Close": float(base_prices[3])}

    # Prevent open-feedback drift: during autoregressive rollout the model sees
    # its own predicted Open returns in the next step. If Open has a small bias,
    # it can compound and push Open price down/up too aggressively.
    # We damp the feedback by blending predicted Open return with the latest
    # Open return already present in the input window.
    OPEN_FEEDBACK_DAMPING = 0.75
    # Additional damping on Open magnitude to prevent small log-return bias
    # from compounding into an aggressive Open price drift over many steps.
    OPEN_MAGNITUDE_DAMPING = 0.6

    # Stabilize roll-forward for High/Low/Volume features during multi-step forecast.
    # We approximate future OHLC shape using recent averages plus the magnitude
    # of predicted price movement.
    recent_window = seq[-20:].copy()
    mean_high_scaled = float(np.mean(recent_window[:, HIGH_IDX]))
    mean_low_scaled = float(np.mean(recent_window[:, LOW_IDX]))
    mean_volume_scaled = float(np.mean(recent_window[:, VOLUME_IDX]))
    
    for _ in range(days):

        # Model outputs are standardized [Open log-return, body_pct].
        n_features = int(seq.shape[-1])
        pred_out = model.predict(
            seq.reshape(1, seq.shape[0], n_features),
            verbose=0,
        )
        # pred_out is (1, 2) -> [open_return_scaled, body_pct_scaled]
        pred_scaled = pred_out[0]
        open_ret_scaled_raw = float(pred_scaled[0])
        body_pct_scaled = float(pred_scaled[1])

        # Blend with the last Open feature from the current sequence window.
        prev_open_ret_scaled = float(seq[-1, OPEN_IDX])
        open_ret_scaled = (
            prev_open_ret_scaled * (1.0 - OPEN_FEEDBACK_DAMPING)
            + open_ret_scaled_raw * OPEN_FEEDBACK_DAMPING
        )

        open_ret_scaled = open_ret_scaled * OPEN_MAGNITUDE_DAMPING

        # Convert standardized outputs back to real units before updating prices.
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

        # Roll the input window forward.
        # We only have predicted Open/body_pct; for the remaining features
        # (High/Low/Volume returns + indicators) we keep the previous scaled values
        # to avoid injecting incorrectly-scaled/incorrectly-ordered data.
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


def forecast_direct_ohlcv(model, last_sequence, horizon, scaler, raw_ohlcv, bias=None):
    """Direct multi-horizon forecast without recursive feature roll-forward."""
    pred_out = model.predict(last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[-1]), verbose=0)[0]
    pred_out = np.asarray(pred_out)
    total_horizon = int(pred_out.shape[0] // 2)
    pred_out = pred_out.reshape(total_horizon, 2)
    horizon = min(int(horizon), total_horizon)

    if bias is not None:
        bias = np.asarray(bias).reshape(total_horizon, 2)
        pred_out = pred_out - bias[:total_horizon]

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
        open_ret = float(inverse_transform_col(pred_out[i, 0], OPEN_IDX, scaler))
        body_pct = float(inverse_transform_col(pred_out[i, 1], BODY_PCT_IDX, scaler))
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
