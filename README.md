# ✍️ AI Handwritten Digit Recognizer - Web Application

An end-to-end Machine Learning web application converted from desktop Tkinter to a modern **Streamlit** platform. The application uses **Scikit-Learn's K-Nearest Neighbors (KNN, $k=3$)** algorithm trained on the built-in **Scikit-Learn Digits Dataset (8×8 resolution, 64 features)** to recognize handwritten digits in real-time.

---

## 🌟 Features

- **🏠 Home Page**: Interactive landing page with project objective, technology stack cards, and pattern recognition fundamentals.
- **✏️ Digit Recognition**: Full-featured digital drawing canvas with mouse/touch input, smart auto-centering, instant KNN prediction, confidence scores ($XX.XX\%$), probability bar chart (0–9), and live 8×8 feature heatmap inspector.
- **🤖 Model Explainer**: In-depth breakdown of the K-Nearest Neighbors algorithm ($k=3$, Euclidean $L_2$ metric), visual pipeline diagram, mathematical formulas, and decision boundaries.
- **📊 Dataset Page**: Live metrics and statistics directly from `sklearn.datasets.load_digits()` (1,797 samples, 10 classes, 64 features, 0–16 pixel range), representative sample gallery (0–9), and interactive dataset browser.
- **ℹ️ About Project**: Comprehensive documentation formatted for college projects, reports, and viva voce examinations (covering problem statement, architecture, advantages, limitations, and future scope).

---

## 📁 Project Structure

```
digit_recognizer/
│
├── app.py                     # Main Streamlit web application
├── requirements.txt           # Python dependency requirements
├── README.md                  # Detailed project documentation & viva guide
└── assets/
    └── style.css              # Glassmorphic custom CSS styling
```

---

## 🚀 Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.9+ (or Python 3.12+) installed.

### 2. Install Dependencies
Run the following command in your terminal:

```bash
pip install -r requirements.txt
```

Or install packages individually:
```bash
pip install streamlit streamlit-drawable-canvas numpy Pillow scikit-learn matplotlib pandas plotly
```

### 3. Run the Web Application
Launch the Streamlit web server:

```bash
streamlit run app.py
```
*(Or if inside `digit_recognizer/` directory: `streamlit run app.py`)*

The application will automatically open in your default browser at:
`http://localhost:8501`

---

## 🧠 Machine Learning Pipeline

1. **Dataset**: Loaded via `sklearn.datasets.load_digits()` (1,797 samples of 8×8 grayscale images).
2. **Model**: `KNeighborsClassifier(n_neighbors=3)` fitted on 64-dimensional feature vectors.
3. **Inference Transformation**:
   - Canvas stroke captured as grayscale.
   - Resized to **8 × 8** using `Image.Resampling.LANCZOS`.
   - Scaled from $[0, 255]$ to $[0, 16]$ matching the training dataset scale:
     $$\text{pixel}_{\text{scaled}} = \left(\frac{\text{pixel}}{255.0}\right) \times 16.0$$
   - Flattened into a 64-element 1D vector.
   - Predicted with `model.predict()` and confidence computed via `model.predict_proba()`.

---

## 🎓 Viva Voce & Theory Quick Reference

- **Q1: Why is KNN called a lazy learner?**
  *Answer:* KNN does not build an explicit mathematical model during training; it memorizes the training instances and delays computation until query time.
- **Q2: Why use $k = 3$ instead of $k = 1$ or an even number?**
  *Answer:* $k=1$ is highly sensitive to noise and outliers. An odd $k=3$ prevents voting ties in binary/multiclass classification.
- **Q3: What distance metric is used?**
  *Answer:* Standard Euclidean Distance ($L_2$ norm) across the 64 feature dimensions:
  $$d(\mathbf{p}, \mathbf{q}) = \sqrt{\sum_{i=1}^{64} (p_i - q_i)^2}$$
- **Q4: Why scale pixel values to 0–16?**
  *Answer:* Scikit-learn's built-in digits dataset records pixel intensity in the range 0 (black background) to 16 (maximum white stroke intensity). Scaling ensures the query input aligns with the training feature space.
