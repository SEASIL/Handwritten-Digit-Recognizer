
# Handwritten-Digit-Recognizer

A full-stack Machine Learning application that trains a Convolutional Neural Network
on the MNIST dataset and lets you draw digits in real-time for instant prediction.

---

## ✨ Features

| Feature | Detail |
|---------|--------|
| **CNN Architecture** | 4× Conv2D · 2× MaxPool · 3× Dropout · Dense 256 → Softmax 10 |
| **Dataset** | MNIST (60 000 train / 10 000 test, 28×28 grayscale) |
| **Accuracy** | ~99.2% test accuracy after 5 epochs |
| **Drawing Canvas** | `streamlit-drawable-canvas` — draw with mouse or touch |
| **Live Prediction** | Real-time digit + confidence distribution chart |
| **UI Theme** | Glassmorphism · Dark · Electric-Blue / Cyberpunk-Purple neons |
| **Charts** | Plotly confidence bar chart + training history line chart |

---

## 🚀 Quick Start

### 1 · Clone / Download

```bash
# If using git
git clone <repo-url>
cd digit_recognizer
```

### 2 · Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows
```

### 3 · Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** TensorFlow may take a few minutes to install.
> On Apple Silicon (M1/M2/M3) use `tensorflow-macos` instead of `tensorflow`.

### 4 · Run the app

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 🏗️ Project Structure

```
digit_recognizer/
├── app.py            ← Main Streamlit application (single-file)
├── requirements.txt  ← Python dependencies
└── README.md         ← This file
```

### Code Sections inside `app.py`

| Section | Description |
|---------|-------------|
| `0. IMPORTS` | All library imports |
| `1. PAGE CONFIG` | Streamlit page metadata |
| `2. DESIGN / CSS` | Glassmorphism + neon CSS injected via `st.markdown` |
| `3. MODEL TRAINING` | `@st.cache_resource` — builds and trains CNN once |
| `4. HELPER FUNCTIONS` | `preprocess_canvas`, `make_confidence_chart`, `make_training_history_chart` |
| `5. MAIN APP LOGIC` | UI layout, canvas, predictions, metrics |

---

## 🧠 CNN Architecture

```
Input: (28, 28, 1)
  │
  ├─ Conv2D 32 × (3,3) ReLU  → (28, 28, 32)
  ├─ Conv2D 32 × (3,3) ReLU  → (28, 28, 32)
  ├─ MaxPooling2D (2,2)       → (14, 14, 32)
  ├─ Dropout 0.25
  │
  ├─ Conv2D 64 × (3,3) ReLU  → (14, 14, 64)
  ├─ Conv2D 64 × (3,3) ReLU  → (14, 14, 64)
  ├─ MaxPooling2D (2,2)       → (7, 7, 64)
  ├─ Dropout 0.25
  │
  ├─ Flatten                  → (3136,)
  ├─ Dense 256 ReLU
  ├─ Dropout 0.5
  └─ Dense 10 Softmax         → (10,)  ← class probabilities
```

**Optimizer:** Adam (lr = 1e-3)  
**Loss:** Categorical Cross-Entropy  
**Epochs:** 5 · **Batch size:** 128

---

## 🖱️ How to Use

1. Wait for the model to finish training (shown in the status box, ~30-90 sec)
2. Draw any digit (0–9) on the black canvas using your mouse
3. The prediction updates instantly in the right panel
4. The **Confidence Distribution** chart shows what the model "thinks" about all 10 digits
5. Click **🗑 CLEAR CANVAS** to reset and draw a new digit

---

## 📦 Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| `tensorflow` | ≥ 2.13 | CNN model |
| `streamlit` | ≥ 1.35 | Web UI framework |
| `streamlit-drawable-canvas` | ≥ 0.9.3 | Freehand drawing widget |
| `plotly` | ≥ 5.18 | Interactive charts |
| `numpy` | ≥ 1.24 | Numerical operations |
| `pandas` | ≥ 2.0 | Architecture table |
| `Pillow` | ≥ 10.0 | Image preprocessing |

All libraries are **100% open-source**.

---

## 🎨 UI Design

The interface uses a **Glassmorphism Cyberpunk** aesthetic:
- **Background:** Deep navy `#060610` with radial purple/blue gradients
- **Cards:** Semi-transparent glass panels with blur and neon borders
- **Accent colours:** Electric Blue `#00d4ff` · Cyberpunk Purple `#9b5de5` · Neon Pink `#f72585`
- **Fonts:** [Orbitron](https://fonts.google.com/specimen/Orbitron) (headings) + [Rajdhani](https://fonts.google.com/specimen/Rajdhani) (body)
- **Overlay:** Subtle scanline texture for extra retro-futurism

---

## 🐛 Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: streamlit_drawable_canvas` | `pip install streamlit-drawable-canvas` |
| Slow first run | Normal — MNIST downloads (~11 MB) and model trains once |
| Canvas not responding | Ensure JavaScript is enabled in your browser |
| Apple Silicon error | Replace `tensorflow` with `tensorflow-macos` in requirements.txt |
>>>>>>> ac928bad (Initial commit of Neural Digit project)
