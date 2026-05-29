# =============================================================================
# SARIP 2026 — IIT Kanpur
# src/models.py
# All Model Architectures: LR, RF, CNN-only, LSTM-only, CNN-LSTM
#
# Author : Shri Harsan M | RA2512052010014 | SRM IST
# Week   : 3–6 (baselines Week 3–4, CNN-LSTM Week 5–6)
# =============================================================================

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ── Reproducibility ───────────────────────────────────────────────────────────
np.random.seed(42)
tf.random.set_seed(42)

# ── Fixed Training Constants — DO NOT CHANGE ──────────────────────────────────
BATCH_SIZE    = 64
EPOCHS        = 50
LEARNING_RATE = 1e-3
PATIENCE      = 7
INPUT_LEN     = 201   # Fixed light curve length


# =============================================================================
# GPU CONFIGURATION — call before any model creation
# =============================================================================
def configure_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"✅ GPU ready: {gpus[0].name}")
    else:
        print("⚠️  No GPU found — running on CPU")


# =============================================================================
# FOCAL LOSS — for class imbalance (γ=2.0, α=0.25)
# =============================================================================
def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """
    Binary focal loss. Down-weights easy examples, focuses on hard ones.
    γ=2.0 and α=0.25 are locked per project spec.

    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
    """
    def focal_loss_fn(y_true, y_pred):
        y_true  = tf.cast(y_true, tf.float32)
        y_pred  = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce     = -y_true * tf.math.log(y_pred) \
                  - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t     = tf.where(tf.equal(tf.cast(y_true, tf.int32), 1),
                           y_pred, 1.0 - y_pred)
        alpha_t = tf.where(tf.equal(tf.cast(y_true, tf.int32), 1),
                           alpha, 1.0 - alpha)
        loss    = alpha_t * tf.pow(1.0 - p_t, gamma) * bce
        return tf.reduce_mean(loss)
    return focal_loss_fn


# =============================================================================
# CUSTOM F1 METRIC — used as early stopping monitor
# =============================================================================
class F1Score(tf.keras.metrics.Metric):
    """
    Binary F1 score as a Keras metric.
    Required because Keras doesn't include F1 natively.
    Used in: EarlyStopping(monitor='val_f1')
    """
    def __init__(self, name: str = 'f1', threshold: float = 0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_bin = tf.cast(y_pred >= self.threshold, tf.float32)
        y_true     = tf.cast(y_true, tf.float32)
        self.tp.assign_add(tf.reduce_sum(y_true * y_pred_bin))
        self.fp.assign_add(tf.reduce_sum((1 - y_true) * y_pred_bin))
        self.fn.assign_add(tf.reduce_sum(y_true * (1 - y_pred_bin)))

    def result(self):
        precision = self.tp / (self.tp + self.fp + 1e-10)
        recall    = self.tp / (self.tp + self.fn + 1e-10)
        return 2.0 * precision * recall / (precision + recall + 1e-10)

    def reset_state(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)


# =============================================================================
# BASELINE 1 — LOGISTIC REGRESSION
# =============================================================================
def build_logistic_regression() -> LogisticRegression:
    """
    Logistic Regression on flattened 201-point input.
    class_weight='balanced' handles the ~1.6:1 class imbalance.
    """
    return LogisticRegression(
        C             = 1.0,
        class_weight  = 'balanced',
        max_iter      = 1000,
        solver        = 'lbfgs',
        random_state  = 42,
        n_jobs        = -1
    )


# =============================================================================
# BASELINE 2 — RANDOM FOREST
# =============================================================================
def build_random_forest() -> RandomForestClassifier:
    """
    Random Forest with 200 trees on flattened 201-point input.
    class_weight='balanced' and fixed random_state for reproducibility.
    """
    return RandomForestClassifier(
        n_estimators  = 200,
        class_weight  = 'balanced',
        max_depth     = None,
        min_samples_split = 2,
        random_state  = 42,
        n_jobs        = -1
    )


# =============================================================================
# BASELINE 3 / ABLATION — CNN-ONLY (1D Convolutional)
# =============================================================================
def build_cnn_only(input_len: int = INPUT_LEN) -> tf.keras.Model:
    """
    Plain 1D-CNN. No LSTM.
    Serves as both a standalone baseline and the CNN ablation variant.

    Architecture:
        Conv1D(64) → BN → MaxPool
        Conv1D(128) → BN → MaxPool
        GlobalAveragePooling1D
        Dense(64) → Dense(1, sigmoid)
    """
    inp = layers.Input(shape=(input_len, 1), name="input")

    # Block 1
    x = layers.Conv1D(64, kernel_size=3, padding='same',
                      activation='relu', name="conv1")(inp)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)

    # Block 2
    x = layers.Conv1D(128, kernel_size=3, padding='same',
                      activation='relu', name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool2")(x)   # ← Grad-CAM target layer

    # Aggregate
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4), name="dense1")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    out = layers.Dense(1, activation='sigmoid', name="output")(x)

    model = models.Model(inputs=inp, outputs=out, name="CNN_only")
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss      = focal_loss(gamma=2.0, alpha=0.25),
        metrics   = [F1Score(name='f1'),
                     tf.keras.metrics.AUC(name='roc_auc'),
                     tf.keras.metrics.Precision(name='precision'),
                     tf.keras.metrics.Recall(name='recall')]
    )
    return model


# =============================================================================
# ABLATION — LSTM-ONLY
# =============================================================================
def build_lstm_only(input_len: int = INPUT_LEN) -> tf.keras.Model:
    """
    LSTM-only model. No convolutional layers.
    Raw normalized input fed directly to stacked LSTMs.

    Architecture:
        LSTM(128, return_sequences=True, dropout=0.3)
        LSTM(64, dropout=0.3)
        Dense(64) → Dense(1, sigmoid)
    """
    inp = layers.Input(shape=(input_len, 1), name="input")

    x = layers.LSTM(128, return_sequences=True,
                    dropout=0.3, name="lstm1")(inp)
    x = layers.LSTM(64, dropout=0.3, name="lstm2")(x)

    x = layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4), name="dense1")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    out = layers.Dense(1, activation='sigmoid', name="output")(x)

    model = models.Model(inputs=inp, outputs=out, name="LSTM_only")
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss      = focal_loss(gamma=2.0, alpha=0.25),
        metrics   = [F1Score(name='f1'),
                     tf.keras.metrics.AUC(name='roc_auc'),
                     tf.keras.metrics.Precision(name='precision'),
                     tf.keras.metrics.Recall(name='recall')]
    )
    return model


# =============================================================================
# MAIN MODEL — CNN-LSTM HYBRID
# =============================================================================
def build_cnn_lstm(input_len: int = INPUT_LEN) -> tf.keras.Model:
    """
    Hybrid CNN-LSTM: convolutional feature extraction followed by
    sequential temporal modelling.

    Architecture (locked per project spec):
        Conv1D(64)  → BN → MaxPool
        Conv1D(128) → BN → MaxPool          ← Grad-CAM target layer
        LSTM(128, return_sequences=True, dropout=0.3)
        LSTM(64, dropout=0.3)
        Dense(64) → Dense(1, sigmoid)

    Loss: Focal Loss (γ=2.0, α=0.25)
    """
    inp = layers.Input(shape=(input_len, 1), name="input")

    # ── CNN feature extraction ────────────────────────────────────────────────
    x = layers.Conv1D(64, kernel_size=3, padding='same',
                      activation='relu', name="conv1")(inp)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)

    x = layers.Conv1D(128, kernel_size=3, padding='same',
                      activation='relu', name="conv2")(x)   # ← Grad-CAM layer
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool2")(x)

    # ── LSTM temporal modelling ───────────────────────────────────────────────
    x = layers.LSTM(128, return_sequences=True,
                    dropout=0.3, name="lstm1")(x)
    x = layers.LSTM(64, dropout=0.3, name="lstm2")(x)

    # ── Classification head ───────────────────────────────────────────────────
    x = layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4), name="dense1")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    out = layers.Dense(1, activation='sigmoid', name="output")(x)

    model = models.Model(inputs=inp, outputs=out, name="CNN_LSTM")
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss      = focal_loss(gamma=2.0, alpha=0.25),
        metrics   = [F1Score(name='f1'),
                     tf.keras.metrics.AUC(name='roc_auc'),
                     tf.keras.metrics.Precision(name='precision'),
                     tf.keras.metrics.Recall(name='recall')]
    )
    return model


# =============================================================================
# MODEL REGISTRY — single lookup used by train.py
# =============================================================================
SKLEARN_MODELS = {
    "lr": build_logistic_regression,
    "rf": build_random_forest,
}

KERAS_MODELS = {
    "cnn"      : build_cnn_only,
    "lstm"     : build_lstm_only,
    "cnn_lstm" : build_cnn_lstm,
}

ALL_MODELS = {**SKLEARN_MODELS, **KERAS_MODELS}


def get_model(name: str):
    """Return a freshly instantiated model by name."""
    if name not in ALL_MODELS:
        raise ValueError(f"Unknown model '{name}'. "
                         f"Choose from: {list(ALL_MODELS.keys())}")
    return ALL_MODELS[name]()


# =============================================================================
# QUICK SANITY CHECK
# =============================================================================
if __name__ == "__main__":
    configure_gpu()
    print("\n── Model Summaries ──────────────────────────────────")

    for name in KERAS_MODELS:
        m = get_model(name)
        print(f"\n{m.name}:")
        m.summary(line_length=65)

    print("\n── Sklearn models ───────────────────────────────────")
    for name in SKLEARN_MODELS:
        m = get_model(name)
        print(f"  {name}: {m}")

    print("\n✅ models.py OK — all architectures instantiated.")
