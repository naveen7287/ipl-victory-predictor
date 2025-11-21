import pickle, os, pprint
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# search for model file
candidates = ["model.pkl", "pipeline.pkl", "model.sav", "model.joblib"]
model_path = None

for c in candidates:
    if os.path.exists(c):
        model_path = c
        break

if model_path is None:
    for root, _, files in os.walk("."):
        for f in files:
            if f.endswith((".pkl", ".sav", ".joblib")):
                model_path = os.path.join(root, f)
                break
        if model_path:
            break

print("\nMODEL FILE:", model_path)

if model_path is None:
    raise SystemExit("No model found !!")

with open(model_path, "rb") as fh:
    model = pickle.load(fh)

print("\nMODEL TYPE:", type(model))

def find_ct(est):
    if isinstance(est, ColumnTransformer):
        return est
    if isinstance(est, Pipeline):
        for name, step in est.steps:
            ct = find_ct(step)
            if ct is not None:
                return ct
    return None

ct = find_ct(model)
print("\nFOUND COLUMN TRANSFORMER:", bool(ct))

if hasattr(ct, "transformers_"):
    print("\n=== COLUMN TRANSFORMERS ===")
    for name, transformer, cols in ct.transformers_:
        print("\n---------------------------------")
        print("NAME:", name)
        print("COLUMNS:", cols)
        print("TRANSFORMER TYPE:", type(transformer))
        print("TRANSFORMER REPR:", repr(transformer))
else:
    print("\nNo transformers_ found. Here is ct:")
    pprint.pprint(ct)
