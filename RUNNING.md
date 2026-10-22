# Running the diamond price demo

Use Python 3.10 or 3.11. From the repository root install `pip install -r requirements.txt`, then run `python app/main.py`. The Flask form is at http://127.0.0.1:5000. The model and encoders load only when a complete prediction is submitted. Numeric inputs must be finite and positive, and categorical inputs must exist in the saved encoders.

The saved model is a historical sales-price regressor, not a current valuation service. The dataset source/license and a reproducible held-out evaluation are not documented. Two local checkouts exist under different remotes; neither has been deleted or overwritten by the other. Runtime artifacts and dependencies are not fetched from user input.
