import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from data_pipeline import prepare_data, OPEN_IDX, CLOSE_IDX
from lstm_model import load_or_train
from evaluation import evaluate_predictions


def main():
    symbol = "AAPL"
    model_path = f"models/{symbol}_lstm_ohlcv_indicators_v9.h5"

    ctx = prepare_data(symbol)

    model, was_loaded = load_or_train(
        ctx["X_train"], ctx["y_train"],
        ctx["X_val"], ctx["y_val"],
        model_path,
    )
    print(f"Model: {'loaded' if was_loaded else 'trained'} -> {model_path}")

    y_pred_both = model.predict(ctx["X_test"], verbose=0)

    m_open, _, _, _ = evaluate_predictions(
        ctx["y_test"][:, 0], y_pred_both[:, 0],
        ctx["scaler"], OPEN_IDX, "Open"
    )
    m_close, _, _, _ = evaluate_predictions(
        ctx["y_test"][:, 1], y_pred_both[:, 1],
        ctx["scaler"], CLOSE_IDX, "Close"
    )

    print("Open:", m_open)
    print("Close:", m_close)


if __name__ == "__main__":
    main()
