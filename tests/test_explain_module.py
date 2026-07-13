"""
Unit tests for the SHAP explainability module.

Ensures the explainability module uses config-provided paths, exposes the
expected public API, and correctly loads data with the configured target
column.
"""

from src.config import FEATURES_PATH, MODEL_PATH, TARGET_COLUMN
from src.explainability.explain import (FIGURES_DIR, MAX_SAMPLES,
                                        create_explainer, generate_shap_values,
                                        load_data, load_model,
                                        plot_feature_importance, plot_summary,
                                        sample_data)


def test_imports():
    assert callable(load_model)
    assert callable(load_data)
    assert callable(sample_data)
    assert callable(create_explainer)
    assert callable(generate_shap_values)
    assert callable(plot_summary)
    assert callable(plot_feature_importance)


def test_constants():
    assert MAX_SAMPLES == 2000
    assert FIGURES_DIR.exists(), f"FIGURES_DIR should exist: {FIGURES_DIR}"


def test_file_existence():
    assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"
    assert FEATURES_PATH.exists(), f"Features not found: {FEATURES_PATH}"


def test_load_functions():
    model = load_model()
    data = load_data()

    assert model is not None
    assert not data.empty
    assert TARGET_COLUMN in data.columns


def test_sampling():
    data = load_data()

    sampled_full = sample_data(data, max_samples=len(data) + 1)
    assert len(sampled_full) == len(data)

    sampled_reduced = sample_data(data, max_samples=100)
    assert 0 < len(sampled_reduced) <= 100


def test_explainer_creation():
    model = load_model()
    data = load_data()
    data_sample = sample_data(data, max_samples=100)
    X = data_sample.drop(columns=[TARGET_COLUMN])

    explainer = create_explainer(model, X)
    assert explainer is not None


def test_module_structure():
    from src.explainability import explain

    required_items = [
        "load_model",
        "load_data",
        "sample_data",
        "create_explainer",
        "generate_shap_values",
        "plot_summary",
        "plot_feature_importance",
        "main",
        "MAX_SAMPLES",
        "FIGURE_DPI",
        "FIGURES_DIR",
    ]

    for item in required_items:
        assert hasattr(explain, item), f"Missing: {item}"


def test_docstrings():
    functions = [
        load_model,
        load_data,
        sample_data,
        create_explainer,
        generate_shap_values,
        plot_summary,
        plot_feature_importance,
    ]

    for func in functions:
        assert func.__doc__ is not None
        assert len(func.__doc__) > 50


def test_type_hints():
    import inspect

    functions = [
        load_model,
        load_data,
        sample_data,
        create_explainer,
        generate_shap_values,
        plot_summary,
        plot_feature_importance,
    ]

    for func in functions:
        sig = inspect.signature(func)
        assert sig.return_annotation != inspect.Signature.empty
