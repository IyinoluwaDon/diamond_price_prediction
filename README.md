# Diamond Sales Price Regression

A Flask application that serves the saved Keras artificial neural network trained for diamond-price regression. The training notebook encodes categorical attributes, selects features with an F-regression score, and splits the data into training and test sets. The network has 24 inputs, 100- and 50-unit ReLU hidden layers, and a linear output; it trains with Adam and mean-squared error for up to 50 epochs. The form validates numeric inputs, uses the saved category mappings, and passes values through the model's preprocessing contract.

`app/model/` contains `tf_m_1.0.0.h5` and explicit category mappings; `scripts/export_preprocessing.py` migrates the notebook's saved encoders into those mappings. `app/main.py` serves the form and prediction route. The app also provides liveness/readiness endpoints and a Gunicorn deployment entry point.

## Run locally

```sh
python -m pip install -r requirements.txt
python app/main.py
```

See `RUNNING.md` for server operation and `tests/` for the request and readiness checks.
