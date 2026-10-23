# Diamond Sales Price Regression Demo

A Flask web form for a saved regression model trained on historical diamond sales data. Inputs are validated and mapped through the preprocessing categories stored with the application.

## What is included

- Form-based prediction with finite numeric and known-category validation
- Lazy model loading and explicit category mappings
- Linux Gunicorn startup and liveness/readiness endpoints

## Run locally

```sh
python -m pip install -r requirements.txt
python app/main.py
```

## Verify

python -m pip install pytest
python -m pytest -q

## Repository map

`app/main.py` serves the Flask app. `app/model/` contains the model and category artifacts. `scripts/` contains the trusted preprocessing-export utility. `RUNNING.md` documents server operation; `tests/` covers requests and readiness.

## Evidence and limitations

Automated request/readiness checks pass, and the saved model has been exercised with a synthetic contract record. That confirms wiring only, not pricing accuracy.

The model estimates historical sales prices and is not a current valuation service. Data provenance, currency/market scope and held-out error are not established. Review training-data and artifact redistribution terms before publication. This checkout is maintained separately from the Banzy checkout.

## License

No repository license file is included. Unless the project owner adds a license, do not assume the code, data or model artifacts may be redistributed or reused. Verify third-party data and model terms separately.
