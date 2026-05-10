# 🚀 Quick Start Guide - SHAP Explainability Module

## Installation & Setup

All dependencies already available in your environment:
```bash
python -m pip list | grep -E "shap|pandas|matplotlib|scikit-learn"
```

✅ **SHAP 0.51.0** is installed
✅ **matplotlib** is available
✅ **pandas** is available

---

## Basic Usage

### Run Standard Analysis
```bash
python -m src.modeling.explain
```

**Output:**
```
reports/figures/
├── shap_summary.png    # SHAP summary plot
└── shap_bar.png        # Feature importance bar chart
```

---

## Programmatic Usage (Python)

### Simple Workflow
```python
from src.modeling.explain import main

# Run with defaults (max 2000 samples)
main()

# Run with custom sample size
main(max_samples=5000)

# Run with custom paths
main(
    max_samples=1000,
    model_path="path/to/model.pkl",
    features_path="path/to/features.parquet"
)
```

### Advanced Workflow
```python
from src.modeling.explain import (
    load_model,
    load_data,
    sample_data,
    create_explainer,
    generate_shap_values,
    plot_summary,
    plot_feature_importance
)

# Load model and data
model = load_model()
data = load_data()

# Sample for performance
X = data.drop(columns=["TARGET"])
data_sample = sample_data(data, max_samples=1000)
X_sample = data_sample.drop(columns=["TARGET"])

# Create SHAP explainer
explainer = create_explainer(model, X)

# Generate SHAP values
shap_values = generate_shap_values(explainer, X_sample)

# Create visualizations
plot_summary(shap_values, X_sample)
plot_feature_importance(shap_values)

# Custom location?
from pathlib import Path
plot_summary(shap_values, X_sample, Path("custom_dir/summary.png"))
```

---

## Graph Interpretation

### Summary Plot (scatter plot with colors)
- **X-axis**: SHAP value (impact on prediction)
  - Left: pushes toward "good" prediction
  - Right: pushes toward "risk" prediction
- **Color**: Feature value
  - Red: high feature value
  - Blue: low feature value
- **Height**: Feature importance ranking

### Bar Plot (horizontal bars)
- **Length**: Mean |SHAP value| (absolute importance)
- **Longer bars**: More important features
- **Order**: Features sorted by importance

---

## Configuration

### Built-in Constants (in explain.py)
```python
MAX_SAMPLES = 2000          # Limit for performance
FIGURE_DPI = 300            # High resolution
FIGURE_FORMAT = "png"       # Output format
REPORTS_DIR = "reports/"    # Output directory
```

### From Config (config.py)
```python
FEATURES_PATH = "data/features/features.parquet"
MODEL_PATH = "models/model.pkl"
TARGET_COLUMN = "TARGET"
RANDOM_STATE = 42
```

---

## File Locations

### Module
```
src/modeling/
└── explain.py             # Main implementation
```

### Documentation
```
reports/
├── README.md             # Comprehensive guide
└── figures/
    ├── shap_summary.png  # Generated plot 1
    └── shap_bar.png      # Generated plot 2
```

### Test
```
test_explain_module.py     # Validation script
```

---

## Troubleshooting

### "Model not found" Error
```
FileNotFoundError: Modelo não encontrado em: models/model.pkl
```
**Solution**: Train model first:
```bash
python -m src.modeling.train
```

### "Features not found" Error
```
FileNotFoundError: Features não encontradas em: data/features/features.parquet
```
**Solution**: Run preprocessing first:
```bash
dvc repro
```

### Memory Issues
**Problem**: Slow execution or out of memory
```python
# Reduce sample size
main(max_samples=500)  # Default is 2000
```

### TreeExplainer not available
**Info**: This is normal
```
⚠ TreeExplainer não disponível, usando KernelExplainer (mais lento)
```
The module will use KernelExplainer instead - slower but works fine.

---

## Integration with DVC Pipeline

To add to your dvc.yaml:
```yaml
stages:
  explain:
    cmd: python -m src.modeling.explain
    deps:
      - models/model.pkl
      - data/features/features.parquet
      - src/modeling/explain.py
    outs:
      - reports/figures/shap_summary.png
      - reports/figures/shap_bar.png
```

Then run:
```bash
dvc repro
```

---

## Logging Output

Every execution logs to stdout in JSON format:
```json
{
  "timestamp": "2026-05-06T21:51:06.205589Z",
  "level": "INFO",
  "logger": "src.modeling.explain",
  "message": "Carregando modelo de: ..."
}
```

Perfect for monitoring and auditing.

---

## Production Deployment

### As a Scheduled Job
```bash
# Cron job example
0 2 * * * cd /path/to/project && python -m src.modeling.explain
```

### As a Docker Container
```dockerfile
RUN python -m src.modeling.explain
```

### As a MLflow Artifact
The module can be logged as an artifact:
```python
import mlflow

with mlflow.start_run():
    main()
    mlflow.log_artifacts("reports/figures/", artifact_path="shap_analysis")
```

---

## Performance Metrics

### Typical Execution Times
- **Load model**: ~0.1s
- **Load features**: ~0.3s
- **Sample 2000 rows**: ~0.3s
- **Create explainer**: ~0.1s (KernelExplainer slower)
- **Generate SHAP values**: ~5-30s (depends on explainer)
- **Create plots**: ~2-5s
- **Total**: ~8-40 seconds

### Memory Usage
- Model: ~10-50 MB
- Features (307k rows): ~60-100 MB
- SHAP values (2k samples): ~5-20 MB
- **Total**: ~100-200 MB

---

## Advanced Topics

### Custom Feature Names
The plots use feature names directly from the DataFrame column names.
To customize:
```python
# Rename before plotting
X_renamed = X.rename(columns={'feature_0': 'Income', 'feature_1': 'Age'})
plot_summary(shap_values, X_renamed)
```

### Export for Reports
```python
from pathlib import Path

# Save high-res for printing
output_dir = Path("reports/figures/")
output_dir.mkdir(parents=True, exist_ok=True)

plot_summary(shap_values, X, output_dir / "summary_hires.png")
plot_feature_importance(shap_values, output_dir / "importance_hires.png")
```

### Compare Models
```python
# Generate explanations for multiple models
models = ['model_v1.pkl', 'model_v2.pkl']
for model_name in models:
    main(model_path=f"models/{model_name}")
```

---

## Documentation References

- **reports/README.md** - Full guide with theory and interpretation
- **src/modeling/explain.py** - Source code with docstrings
- **IMPLEMENTATION_SUMMARY.md** - Technical details
- **test_explain_module.py** - Example usage patterns

---

## Support

For questions about SHAP:
- [Official Docs](https://shap.readthedocs.io/)
- [GitHub Issues](https://github.com/slundberg/shap/issues)

For questions about this implementation:
- Read docstrings: `python -c "from src.modeling.explain import *; help(main)"`
- Check test script: `python test_explain_module.py`
- Review comments in explain.py

---

**Status**: ✅ Production-Ready  
**Version**: 1.0  
**Last Updated**: 2026-05-06
