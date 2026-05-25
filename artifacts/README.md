# Model Artifacts

Run the export script from this directory's parent to create the local model files:

```powershell
python dump_model.py --ratings combined_data_1.txt --user-id 1331154
```

The script writes:

- `svd_model.joblib`: joblib-saved Surprise `SVD` model.
- `predictions_user_<id>.json`: ranked predictions consumed by `server.js`.
- `model_manifest.json`: metadata for the latest export.
