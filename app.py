"""
╔══════════════════════════════════════════════════════════════════════╗
║          NEURAL DIGIT — Handwritten Digit Recognizer                ║
║          CNN on MNIST | Streamlit + TensorFlow/Keras                ║
╚══════════════════════════════════════════════════════════════════════╝

Architecture:
  - 2× Conv2D → MaxPooling → Dropout → Dense → Softmax
  - Trained on MNIST (60 000 samples, 10 classes)
  - ~99% test-set accuracy after 5 epochs

UI:
  - Glassmorphism dark theme with Electric-Blue / Cyberpunk-Purple neons
  - streamlit-drawable-canvas for freehand digit input
  - Plotly confidence-distribution chart (live)
  - Multi-column layout with live model metrics
"""

# ─────────────────────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────────────────────
import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image, ImageOps

# ─────────────────────────────────────────────────────────────
# 1. PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Neural Digit — CNN Recognizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# 2. DESIGN / CSS  — Glassmorphism + Cyberpunk Neon Theme
# ─────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
/* ── Google Fonts ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&display=swap');

/* ── CSS Variables ────────────────────────────────────────── */
:root {
  --bg-deep:      #060610;
  --bg-mid:       #0d0d1a;
  --glass-bg:     rgba(13, 13, 40, 0.6);
  --glass-border: rgba(100, 80, 255, 0.25);
  --neon-blue:    #00d4ff;
  --neon-purple:  #9b5de5;
  --neon-pink:    #f72585;
  --neon-green:   #06ffa5;
  --text-primary: #e8e8ff;
  --text-dim:     #7878aa;
  --glow-blue:    0 0 20px rgba(0, 212, 255, 0.5);
  --glow-purple:  0 0 20px rgba(155, 93, 229, 0.5);
}

/* ── Global Reset ─────────────────────────────────────────── */
html, body, [class*="css"] {
  font-family: 'Rajdhani', sans-serif;
  color: var(--text-primary);
}

/* Deep animated background */
.stApp {
  background: var(--bg-deep);
  background-image:
    radial-gradient(ellipse 80% 50% at 20% 30%, rgba(155,93,229,0.07) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 80% 70%, rgba(0,212,255,0.07) 0%, transparent 60%);
}

/* ── Glassmorphism Card ───────────────────────────────────── */
.glass-card {
  background: var(--glass-bg);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid var(--glass-border);
  border-radius: 16px;
  padding: 24px 28px;
  margin-bottom: 20px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.06);
}

/* ── Hero Title ───────────────────────────────────────────── */
.hero-title {
  font-family: 'Orbitron', sans-serif;
  font-size: 2.8rem;
  font-weight: 900;
  letter-spacing: 4px;
  background: linear-gradient(135deg, var(--neon-blue) 0%, var(--neon-purple) 50%, var(--neon-pink) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  text-align: center;
  margin-bottom: 4px;
  text-shadow: none;
  filter: drop-shadow(0 0 30px rgba(0,212,255,0.4));
}

.hero-subtitle {
  font-family: 'Rajdhani', sans-serif;
  font-size: 1.0rem;
  font-weight: 300;
  color: var(--text-dim);
  text-align: center;
  letter-spacing: 6px;
  text-transform: uppercase;
  margin-bottom: 36px;
}

/* ── Section Headers ──────────────────────────────────────── */
.section-label {
  font-family: 'Orbitron', sans-serif;
  font-size: 0.7rem;
  letter-spacing: 4px;
  text-transform: uppercase;
  color: var(--neon-blue);
  margin-bottom: 10px;
  padding-bottom: 6px;
  border-bottom: 1px solid rgba(0,212,255,0.2);
}

/* ── Prediction Badge ─────────────────────────────────────── */
.prediction-badge {
  font-family: 'Orbitron', sans-serif;
  font-size: 6rem;
  font-weight: 900;
  text-align: center;
  line-height: 1;
  background: linear-gradient(135deg, var(--neon-blue), var(--neon-purple));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  filter: drop-shadow(0 0 40px rgba(0,212,255,0.6));
  padding: 10px 0;
}

.confidence-label {
  font-family: 'Rajdhani', sans-serif;
  font-size: 0.85rem;
  letter-spacing: 3px;
  color: var(--neon-green);
  text-align: center;
  text-transform: uppercase;
}

/* ── Metric Tiles ─────────────────────────────────────────── */
.metric-tile {
  background: rgba(0,212,255,0.05);
  border: 1px solid rgba(0,212,255,0.15);
  border-radius: 12px;
  padding: 16px 20px;
  text-align: center;
}
.metric-value {
  font-family: 'Orbitron', sans-serif;
  font-size: 1.6rem;
  font-weight: 700;
  color: var(--neon-blue);
}
.metric-name {
  font-size: 0.7rem;
  letter-spacing: 2px;
  color: var(--text-dim);
  text-transform: uppercase;
}

/* ── Neon Divider ─────────────────────────────────────────── */
.neon-divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--neon-purple), var(--neon-blue), transparent);
  margin: 24px 0;
  opacity: 0.6;
}

/* ── Canvas container ─────────────────────────────────────── */
.canvas-wrapper {
  border: 1px solid rgba(155,93,229,0.3);
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 0 30px rgba(155,93,229,0.15), inset 0 0 40px rgba(0,0,0,0.3);
}

/* ── Status / Spinner override ────────────────────────────── */
.stStatus, .stSpinner > div {
  background: var(--glass-bg) !important;
  border: 1px solid var(--glass-border) !important;
  border-radius: 12px !important;
}

/* ── Streamlit element cleanup ────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; max-width: 1200px; }

/* ── Button ───────────────────────────────────────────────── */
.stButton > button {
  background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(155,93,229,0.15));
  border: 1px solid rgba(0,212,255,0.4);
  color: var(--neon-blue);
  font-family: 'Orbitron', sans-serif;
  font-size: 0.7rem;
  letter-spacing: 3px;
  border-radius: 8px;
  transition: all 0.3s ease;
  width: 100%;
}
.stButton > button:hover {
  background: linear-gradient(135deg, rgba(0,212,255,0.3), rgba(155,93,229,0.3));
  box-shadow: var(--glow-blue);
  border-color: var(--neon-blue);
  transform: translateY(-1px);
}

/* ── Info / warning banners ───────────────────────────────── */
.stAlert {
  background: rgba(0,212,255,0.06) !important;
  border: 1px solid rgba(0,212,255,0.2) !important;
  border-radius: 10px !important;
  color: var(--text-primary) !important;
}

/* Scanline overlay for extra cyberpunk feel */
.stApp::after {
  content: "";
  position: fixed;
  inset: 0;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0,0,0,0.03) 2px,
    rgba(0,0,0,0.03) 4px
  );
  pointer-events: none;
  z-index: 9999;
}
</style>
"""

# ─────────────────────────────────────────────────────────────
# 3. MODEL TRAINING  (cached — runs only once per session)
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def build_and_train_model():
    """
    Build a 2-block CNN, train on MNIST, and return:
      (model, history_dict, test_accuracy, train_time_seconds)
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    # ── Load & Pre-process MNIST ──────────────────────────────
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalise to [0, 1] and add channel dim → (N, 28, 28, 1)
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test  = np.expand_dims(x_test,  -1)

    # One-hot encode labels
    y_train_oh = tf.keras.utils.to_categorical(y_train, 10)
    y_test_oh  = tf.keras.utils.to_categorical(y_test,  10)

    # ── CNN Architecture ─────────────────────────────────────
    model = models.Sequential([
        # Block 1 — feature extraction
        layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                      input_shape=(28, 28, 1), name="conv1_a"),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv1_b"),
        layers.MaxPooling2D((2, 2), name="pool1"),
        layers.Dropout(0.25, name="drop1"),

        # Block 2 — deeper features
        layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2_a"),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2_b"),
        layers.MaxPooling2D((2, 2), name="pool2"),
        layers.Dropout(0.25, name="drop2"),

        # Classifier head
        layers.Flatten(name="flatten"),
        layers.Dense(256, activation="relu", name="dense1"),
        layers.Dropout(0.5, name="drop3"),
        layers.Dense(10, activation="softmax", name="output"),
    ], name="NeuralDigit_CNN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # ── Training ─────────────────────────────────────────────
    t0 = time.time()
    history = model.fit(
        x_train, y_train_oh,
        epochs=5,
        batch_size=128,
        validation_split=0.1,
        verbose=0,
    )
    train_time = round(time.time() - t0, 1)

    # ── Evaluation ───────────────────────────────────────────
    _, test_acc = model.evaluate(x_test, y_test_oh, verbose=0)

    return model, history.history, round(test_acc * 100, 2), train_time


# ─────────────────────────────────────────────────────────────
# 4. HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def preprocess_canvas(canvas_image_data: np.ndarray) -> np.ndarray | None:
    """
    Convert the RGBA canvas numpy array → (1, 28, 28, 1) float32 tensor.
    Centers the digit and scales it to fit in a 20x20 box inside the 28x28 image,
    matching the MNIST dataset preprocessing for robust predictions.
    Returns None if the canvas appears to be blank.
    """
    if canvas_image_data is None:
        return None

    # RGBA → PIL Image (RGB)
    img = Image.fromarray(canvas_image_data.astype("uint8"), mode="RGBA")
    img = img.convert("L")                        # grayscale

    # If almost all pixels are dark/empty → blank canvas
    arr = np.array(img)
    if arr.max() < 30:
        return None

    # Extract the bounding box of the drawn digit
    coords = np.argwhere(arr > 30)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop the image to the bounding box
    cropped = arr[y_min:y_max+1, x_min:x_max+1]

    # Make it square to preserve aspect ratio
    h, w = cropped.shape
    max_dim = max(h, w)
    pad_y = (max_dim - h) // 2
    pad_y_extra = (max_dim - h) - pad_y
    pad_x = (max_dim - w) // 2
    pad_x_extra = (max_dim - w) - pad_x

    squared = np.pad(
        cropped, 
        ((pad_y, pad_y_extra), (pad_x, pad_x_extra)), 
        mode='constant', 
        constant_values=0
    )

    # Resize to 20x20 like MNIST digits
    img_sq = Image.fromarray(squared).resize((20, 20), Image.LANCZOS)

    # Pad with 4 pixels on all sides to make it 28x28 (MNIST standard)
    img_28 = np.pad(np.array(img_sq), ((4, 4), (4, 4)), mode='constant', constant_values=0)

    # Normalise
    tensor = img_28.astype("float32") / 255.0
    tensor = np.expand_dims(tensor, axis=(0, -1))  # (1, 28, 28, 1)
    return tensor


def make_confidence_chart(probs: np.ndarray) -> go.Figure:
    """
    Build a horizontal Plotly bar chart showing P(digit) for 0–9.
    """
    digits     = [str(i) for i in range(10)]
    values     = (probs * 100).tolist()
    top_digit  = int(np.argmax(probs))

    bar_colors = [
        "rgba(247,37,133,0.85)" if i == top_digit else "rgba(0,212,255,0.45)"
        for i in range(10)
    ]
    border_colors = [
        "#f72585" if i == top_digit else "rgba(0,212,255,0.6)"
        for i in range(10)
    ]

    fig = go.Figure(go.Bar(
        x=values,
        y=digits,
        orientation="h",
        marker=dict(
            color=bar_colors,
            line=dict(color=border_colors, width=1.5),
        ),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(family="Rajdhani", size=11, color="#e8e8ff"),
        hovertemplate="Digit %{y}: %{x:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=60, t=10, b=10),
        height=280,
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            tickfont=dict(family="Rajdhani", color="#7878aa", size=10),
            range=[0, 105],
            zeroline=False,
        ),
        yaxis=dict(
            tickfont=dict(family="Orbitron", color="#e8e8ff", size=12),
            autorange="reversed",
        ),
        hoverlabel=dict(bgcolor="#0d0d1a", font_family="Rajdhani"),
    )
    return fig


def make_training_history_chart(history: dict) -> go.Figure:
    """
    Plot training vs validation accuracy across epochs.
    """
    epochs = list(range(1, len(history["accuracy"]) + 1))

    fig = go.Figure()

    # Training accuracy
    fig.add_trace(go.Scatter(
        x=epochs, y=[v * 100 for v in history["accuracy"]],
        mode="lines+markers",
        name="Train Acc",
        line=dict(color="#00d4ff", width=2),
        marker=dict(size=6, color="#00d4ff",
                    line=dict(color="#060610", width=1)),
    ))

    # Validation accuracy
    fig.add_trace(go.Scatter(
        x=epochs, y=[v * 100 for v in history["val_accuracy"]],
        mode="lines+markers",
        name="Val Acc",
        line=dict(color="#9b5de5", width=2, dash="dash"),
        marker=dict(size=6, color="#9b5de5",
                    line=dict(color="#060610", width=1)),
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=180,
        legend=dict(
            font=dict(family="Rajdhani", color="#e8e8ff", size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            title=dict(text="Epoch", font=dict(family="Rajdhani",
                                               color="#7878aa", size=11)),
            tickfont=dict(family="Rajdhani", color="#7878aa", size=10),
            showgrid=True, gridcolor="rgba(255,255,255,0.05)",
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text="Accuracy (%)", font=dict(family="Rajdhani",
                                                      color="#7878aa", size=11)),
            tickfont=dict(family="Rajdhani", color="#7878aa", size=10),
            showgrid=True, gridcolor="rgba(255,255,255,0.05)",
            zeroline=False,
        ),
        hoverlabel=dict(bgcolor="#0d0d1a", font_family="Rajdhani"),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# 5. MAIN APP LOGIC
# ─────────────────────────────────────────────────────────────

def main():
    if "canvas_key" not in st.session_state:
        st.session_state["canvas_key"] = "canvas_0"

    # ── Inject CSS ────────────────────────────────────────────
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Hero Header ───────────────────────────────────────────
    st.markdown('<h1 class="hero-title">NEURAL DIGIT</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">Convolutional Neural Network · MNIST · Real-Time Recognition</p>',
        unsafe_allow_html=True,
    )

    # ── Load / Train Model (with animated status) ─────────────
    with st.status("⚡ Initialising Neural Network…", expanded=True) as status:
        st.write("🔬 Loading MNIST dataset (70 000 samples)…")
        time.sleep(0.4)
        st.write("🧠 Building CNN architecture (Conv2D × 4 + Dense)…")
        time.sleep(0.3)
        st.write("🚀 Training on 54 000 samples — 5 epochs…")
        model, history, test_acc, train_time = build_and_train_model()
        st.write(f"✅ Training complete!  Test accuracy → **{test_acc}%**")
        status.update(label=f"✅ Model ready — {test_acc}% accuracy", state="complete")

    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    # ── Two-column layout ─────────────────────────────────────
    col_left, col_right = st.columns([1.1, 1], gap="large")

    # ════════════════════════════════════════════════════════
    # LEFT COLUMN — Drawing Canvas
    # ════════════════════════════════════════════════════════
    with col_left:
        st.markdown('<div class="section-label">✏️ Draw a Digit (0 – 9)</div>',
                    unsafe_allow_html=True)

        try:
            from streamlit_drawable_canvas import st_canvas

            st.markdown('<div class="canvas-wrapper">', unsafe_allow_html=True)
            canvas_result = st_canvas(
                fill_color="rgba(0,0,0,0)",
                stroke_width=18,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=340,
                width=340,
                drawing_mode="freedraw",
                key=st.session_state["canvas_key"],
                display_toolbar=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        except ImportError:
            st.warning("⚠️  `streamlit-drawable-canvas` not installed.\n\n"
                       "Run:  `pip install streamlit-drawable-canvas`")
            canvas_result = None

        # Clear button
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑  CLEAR CANVAS", key="clear_btn"):
            st.session_state["canvas_key"] = f"canvas_{time.time()}"
            st.rerun()

        # Model architecture overview
        st.markdown('<br><div class="section-label">🏗️ CNN Architecture</div>',
                    unsafe_allow_html=True)
        arch_data = {
            "Layer": [
                "Conv2D 32×3×3", "Conv2D 32×3×3", "MaxPool + Dropout",
                "Conv2D 64×3×3", "Conv2D 64×3×3", "MaxPool + Dropout",
                "Dense 256",     "Dropout 0.5",   "Dense 10 (Softmax)",
            ],
            "Output Shape": [
                "28×28×32", "28×28×32", "14×14×32",
                "14×14×64", "14×14×64", "7×7×64",
                "256",       "256",      "10",
            ],
        }
        st.dataframe(
            pd.DataFrame(arch_data),
            use_container_width=True,
            hide_index=True,
        )

    # ════════════════════════════════════════════════════════
    # RIGHT COLUMN — Prediction & Metrics
    # ════════════════════════════════════════════════════════
    with col_right:

        # ── Prediction Section ────────────────────────────────
        st.markdown('<div class="section-label">🎯 Prediction</div>',
                    unsafe_allow_html=True)

        image_data = getattr(canvas_result, "image_data", None) \
            if canvas_result is not None else None
        tensor = preprocess_canvas(image_data)

        if tensor is not None:
            probs      = model.predict(tensor, verbose=0)[0]
            pred_digit = int(np.argmax(probs))
            confidence = float(probs[pred_digit]) * 100

            # Big predicted digit badge
            st.markdown(
                f'<div class="prediction-badge">{pred_digit}</div>'
                f'<div class="confidence-label">Confidence: {confidence:.1f}%</div>',
                unsafe_allow_html=True,
            )

            st.markdown('<br>', unsafe_allow_html=True)
            st.markdown('<div class="section-label">📊 Confidence Distribution</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(
                make_confidence_chart(probs),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        else:
            # Placeholder when canvas is blank
            st.markdown(
                '<div style="text-align:center; padding: 60px 0;">'
                '<div style="font-size:4rem;">✍️</div>'
                '<div style="font-family:Rajdhani;color:#7878aa;letter-spacing:2px;'
                'font-size:0.9rem;margin-top:12px;">DRAW A DIGIT TO BEGIN</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

        # ── Model Metrics ─────────────────────────────────────
        st.markdown('<div class="section-label">📈 Model Metrics</div>',
                    unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f'<div class="metric-tile">'
                f'<div class="metric-value">{test_acc}%</div>'
                f'<div class="metric-name">Test Accuracy</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with m2:
            val_acc_last = round(history["val_accuracy"][-1] * 100, 2)
            st.markdown(
                f'<div class="metric-tile">'
                f'<div class="metric-value">{val_acc_last}%</div>'
                f'<div class="metric-name">Val Accuracy</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="metric-tile">'
                f'<div class="metric-value">{train_time}s</div>'
                f'<div class="metric-name">Train Time</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">📉 Training History</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            make_training_history_chart(history),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # ── Footer ────────────────────────────────────────────────
    st.markdown(
        '<div style="text-align:center;padding:30px 0 10px;'
        'font-family:Rajdhani;font-size:0.75rem;color:#3a3a6a;letter-spacing:3px;">'
        'NEURAL DIGIT &nbsp;·&nbsp; CNN × MNIST &nbsp;·&nbsp; '
        'TensorFlow · Streamlit · Plotly'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
