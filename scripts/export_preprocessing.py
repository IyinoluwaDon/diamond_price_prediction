"""One-time migration of trusted bundled LabelEncoders to explicit category lists."""
from pathlib import Path
import json
import joblib

root=Path(__file__).resolve().parents[1]/"app/model"
categories={path.stem:[str(value) for value in joblib.load(path).classes_] for path in root.glob("*.joblib")}
if len(categories)!=18:raise ValueError("Expected 18 categorical encoders")
(root/"categories.json").write_text(json.dumps(categories,indent=2),encoding="utf-8")
print("Exported 18 category mappings in their original class order")
