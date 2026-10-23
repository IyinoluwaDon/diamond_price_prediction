# Running the diamond price demo

Use Python 3.10 or 3.11. From the repository root install `pip install -r requirements.txt`, then run `python app/main.py`. The Flask form is at http://127.0.0.1:5000. The model and encoders load only when a complete prediction is submitted. Numeric inputs must be finite and positive, and categorical inputs must exist in the saved encoders.

The saved model is a historical sales-price regressor, not a current valuation service. The dataset source/license and a reproducible held-out evaluation are not documented. Two local checkouts exist under different remotes; neither has been deleted or overwritten by the other. Runtime artifacts and dependencies are not fetched from user input.

Runtime category mappings are stored explicitly in `app/model/categories.json`, exported in the original LabelEncoder class order. No sklearn pickle is loaded during inference. The historical joblib encoders remain for provenance. Real saved-model inference was exercised with a complete synthetic contract record; this is not a price-accuracy benchmark.

To reproduce the one-time trusted-artifact migration, install `joblib` and `scikit-learn`, then run `python scripts/export_preprocessing.py`. The review used sklearn 1.9.1 to read class arrays from the older 1.0.2 encoders; prediction no longer depends on pickle compatibility. Never run this script against untrusted pickle/joblib files.
