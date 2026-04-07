# Diamond Sales Price Prediction

A deep learning web application that predicts the sales price of a diamond based on its physical and grading characteristics. Users input 24 diamond attributes through a Flask web interface and receive an instant price estimate from a trained TensorFlow neural network.

---

## Overview

This project trains a regression model on a dataset of over **219,000 diamond sales records** to predict `total_sales_price`. Categorical features (cut, color, clarity, etc.) are label-encoded using pre-fitted `joblib` encoders, and numerical features are passed directly into the model. The trained neural network (`tf_m_1.0.0.h5`) handles inference at serving time.

---

## Project Structure

```
Project Code/
│
├── app/
│   ├── main.py                         # Flask app — routes and form handling
│   ├── requirements.txt                # Python dependencies
│   ├── model/
│   │   ├── model.py                    # Model loading, encoding, prediction pipeline
│   │   ├── tf_m_1.0.0.h5              # Trained TensorFlow/Keras model
│   │   ├── cut.joblib                  # Label encoder — cut
│   │   ├── color.joblib                # Label encoder — color
│   │   ├── clarity.joblib              # Label encoder — clarity
│   │   ├── cut_quality.joblib          # Label encoder — cut quality
│   │   ├── lab.joblib                  # Label encoder — grading lab
│   │   ├── symmetry.joblib             # Label encoder — symmetry
│   │   ├── polish.joblib               # Label encoder — polish
│   │   ├── eye_clean.joblib            # Label encoder — eye clean
│   │   ├── culet_size.joblib           # Label encoder — culet size
│   │   ├── culet_condition.joblib      # Label encoder — culet condition
│   │   ├── girdle_min.joblib           # Label encoder — minimum girdle
│   │   ├── girdle_max.joblib           # Label encoder — maximum girdle
│   │   ├── fluor_color.joblib          # Label encoder — fluorescence color
│   │   ├── fluor_intensity.joblib      # Label encoder — fluorescence intensity
│   │   ├── fancy_color_dominant_color.joblib
│   │   ├── fancy_color_secondary_color.joblib
│   │   ├── fancy_color_overtone.joblib
│   │   └── fancy_color_intensity.joblib
│   └── templates/
│       ├── index.html                  # Main input form and prediction display
│       └── sub.html                    # Auxiliary template
│
└── ML_model/
    ├── diamondSalesReg.ipynb           # Model training notebook
    ├── diamonds.csv                    # Raw dataset (~219,000 rows)
    └── tf_m_1.0.0.h5                  # Model checkpoint (training copy)
```

---

## Dataset

**File:** `diamonds.csv`  
**Records:** ~219,000 diamond sales transactions  
**Price range:** $200 – $1,449,881

### Features

| Feature | Type | Description |
|---------|------|-------------|
| `cut` | Categorical | Diamond shape (Round, Pear, Oval, Emerald, Princess, Marquise, Heart, Cushion, Radiant, Asscher) |
| `color` | Categorical | Color grade (D through L scale) |
| `clarity` | Categorical | Clarity grade (IF, VVS1, VVS2, VS1, VS2, SI1, SI2, I1, I2) |
| `carat_weight` | Numerical | Weight of the diamond in carats |
| `cut_quality` | Categorical | Quality of the cut (e.g. Excellent, Very Good) |
| `lab` | Categorical | Grading laboratory (GIA, IGI, HRD) |
| `symmetry` | Categorical | Symmetry grade |
| `polish` | Categorical | Polish grade |
| `eye_clean` | Categorical | Whether inclusions are visible to the naked eye |
| `culet_size` | Categorical | Size of the culet facet |
| `culet_condition` | Categorical | Condition of the culet |
| `depth_percent` | Numerical | Total depth percentage |
| `table_percent` | Numerical | Table size as a percentage of diameter |
| `meas_length` | Numerical | Diamond length (mm) |
| `meas_width` | Numerical | Diamond width (mm) |
| `meas_depth` | Numerical | Diamond depth (mm) |
| `girdle_min` | Categorical | Minimum girdle thickness |
| `girdle_max` | Categorical | Maximum girdle thickness |
| `fluor_color` | Categorical | Color of fluorescence |
| `fluor_intensity` | Categorical | Intensity of fluorescence |
| `fancy_color_dominant_color` | Categorical | Primary fancy color (if applicable) |
| `fancy_color_secondary_color` | Categorical | Secondary fancy color (if applicable) |
| `fancy_color_overtone` | Categorical | Fancy color overtone (if applicable) |
| `fancy_color_intensity` | Categorical | Fancy color intensity (if applicable) |

**Target:** `total_sales_price` (USD)

---

## Model

**Architecture:** TensorFlow / Keras Neural Network (regression)  
**Loss function:** Mean Squared Error  
**Optimizer:** Adam  
**Serialization:** HDF5 (`.h5`)  
**Categorical encoding:** Scikit-learn `LabelEncoder`, one per categorical feature, saved as `.joblib` files

### Prediction Pipeline (`model.py`)

1. Accepts a list of 24 raw feature values (mixed types).
2. Wraps them into a single-row `DataFrame` with named columns.
3. For each categorical column, loads the corresponding pre-fitted `LabelEncoder` from disk and transforms the value.
4. Casts the full row to `float64` and runs inference through the neural network.
5. Returns the predicted sales price as a scalar.

---

## Web Application

Built with **Flask**. The app renders a 24-field input form, collects diamond attributes on `POST`, runs the prediction pipeline, and displays the estimated price on the same page.

### Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/` | `GET` | Renders the input form |
| `/` | `POST` | Accepts all 24 features, predicts price, renders result |

---

## Setup & Usage

### Prerequisites

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

> **Note:** `requirements.txt` does not include TensorFlow or Keras. Install them separately:
> ```bash
> pip install tensorflow
> ```

Full dependency list:

```
flask==2.2.2
numpy==1.23.5
pandas==1.5.3
scikit-learn==1.2.1
tensorflow  # not in requirements.txt — install manually
```

### Run the App

```bash
cd "Project Code/app"
python main.py
```

Then open `http://127.0.0.1:5000` in your browser.

### Making a Prediction

Fill in all 24 fields in the form and click **Submit**. The predicted sales price will appear below the form.

**Example inputs:**

| Field | Example Value |
|-------|--------------|
| cut | `Round` |
| color | `E` |
| clarity | `VVS1` |
| carat_weight | `0.5` |
| cut_quality | `Excellent` |
| lab | `GIA` |
| symmetry | `Very Good` |
| polish | `Excellent` |
| depth_percent | `61.5` |
| table_percent | `57.0` |
| meas_length | `5.10` |
| meas_width | `5.09` |
| meas_depth | `3.13` |

---

## Notebook

`ML_model/diamondSalesReg.ipynb` contains the full model development workflow:

- Data loading and exploratory analysis on `diamonds.csv`
- Categorical feature encoding with `LabelEncoder`
- Neural network architecture definition and training
- Model evaluation and export to `.h5`
- Encoder export to individual `.joblib` files

---

## Limitations

- Categorical inputs are **case-sensitive** and must exactly match the values seen during training. Invalid strings will raise a `LabelEncoder` transform error.
- `tensorflow` and `keras` are missing from `requirements.txt` — they must be installed separately.
- The form provides no input validation or dropdown guidance, so users must know valid category values in advance.
- Fancy color fields should be set to `"unknown"` for standard (non-fancy) diamonds.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Model | TensorFlow / Keras (Neural Network Regressor) |
| Feature Encoding | Scikit-learn `LabelEncoder` |
| Data Processing | Pandas, NumPy |
| Web Framework | Flask |
| Model Serialization | HDF5 (`.h5`), Joblib (`.joblib`) |
| Frontend | HTML |
