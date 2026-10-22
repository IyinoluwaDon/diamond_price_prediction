from pathlib import Path
from functools import lru_cache
import math
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
features = ['cut', 'color', 'clarity', 'carat_weight', 'cut_quality', 'lab', 'symmetry', 'polish', 'eye_clean', 'culet_size', 'culet_condition', 'depth_percent', 'table_percent', 'meas_length', 'meas_width', 'meas_depth', 'girdle_min', 'girdle_max', 'fluor_color', 'fluor_intensity', 'fancy_color_dominant_color', 'fancy_color_secondary_color', 'fancy_color_overtone', 'fancy_color_intensity']
numeric = ['carat_weight', 'depth_percent', 'table_percent', 'meas_length', 'meas_width', 'meas_depth']

@lru_cache(maxsize=1)
def artifacts():
    try:
        import joblib
        from tensorflow.keras.models import load_model
        model = load_model(BASE_DIR / 'tf_m_1.0.0.h5', compile=False)
        encoders = {key: joblib.load(BASE_DIR / f'{key}.joblib') for key in features if key not in numeric}
        return model, encoders
    except Exception as exc:
        raise RuntimeError('Model or encoders could not be loaded. Install requirements.txt and use the bundled artifacts.') from exc

def predict_pipe(values):
    if len(values) != len(features):
        raise ValueError('All 24 diamond attributes are required.')
    converted = dict(zip(features, values))
    for key in numeric:
        try:
            converted[key] = float(converted[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{key} must be a number.') from exc
        if not math.isfinite(converted[key]) or converted[key] <= 0:
            raise ValueError(f'{key} must be a finite positive number.')
    model, encoders = artifacts()
    for key, encoder in encoders.items():
        value = str(converted[key]).strip()
        if value not in encoder.classes_:
            raise ValueError(f'Unsupported value for {key}: choose one of the training categories.')
        converted[key] = int(encoder.transform([value])[0])
    row = np.asarray([[converted[key] for key in features]], dtype=np.float32)
    try:
        price = float(np.asarray(model(row, training=False)).reshape(-1)[0])
    except Exception as exc:
        raise RuntimeError('The saved model could not run inference.') from exc
    if not math.isfinite(price) or price < 0:
        raise RuntimeError('The model returned an invalid price.')
    return round(price, 2)
