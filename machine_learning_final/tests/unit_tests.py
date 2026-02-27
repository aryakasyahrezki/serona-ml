"""
unit_tests.py

Purpose: Pytest suite for validating the face shape classification system.
         Covers artifact structure, inference behavior, and API implementation integrity.

Run with:
    pytest machine_learning_final/tests/unit_tests.py -v
"""

import os
import pytest
import joblib
import numpy as np
import pandas as pd

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/model.pkl')
API_PATH = os.path.join(os.path.dirname(__file__), '../scripts/api.py')

EXPECTED_CLASSES = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
EXPECTED_FEATURES = [
    'ratio_len_width', 'ratio_jaw_cheek', 'ratio_forehead_jaw',
    'avg_jaw_angle', 'ratio_chin_jaw', 'circularity', 'solidity', 'extent'
]
MIN_CV_F1_MACRO = 0.65  # Changed from MIN_CV_ACCURACY
EXPECTED_SEED = 4  # Changed from 47

# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture(scope='module')
def artifact():
    """Load model artifact once, shared across all tests."""
    assert os.path.exists(MODEL_PATH), f"model.pkl not found at {MODEL_PATH}"
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope='module')
def pipeline(artifact):
    """Return the sklearn pipeline."""
    return artifact['pipeline']


@pytest.fixture(scope='module')
def label_encoder(artifact):
    """Return the label encoder."""
    return artifact['label_encoder']


@pytest.fixture(scope='module')
def feature_names(artifact):
    """Return list of feature names."""
    return artifact['feature_names']


@pytest.fixture(scope='module')
def dummy_sample(feature_names):
    """Return a realistic dummy input sample as DataFrame."""
    return pd.DataFrame([{
        'ratio_len_width': 1.18,
        'ratio_jaw_cheek': 0.88,
        'ratio_forehead_jaw': 0.85,
        'avg_jaw_angle': 145.0,
        'ratio_chin_jaw': 0.88,
        'circularity': 0.95,
        'solidity': 0.999,
        'extent': 0.79
    }], columns=feature_names)

# ==========================================
# 1. MODEL STRUCTURE TESTS
# ==========================================

class TestModelStructure:
    """Ensure the saved model artifact contains proper structure and required metadata."""

    def test_model_file_exists(self):
        """Model file must exist at expected path."""
        assert os.path.exists(MODEL_PATH), "model.pkl not found"

    def test_required_keys_present(self, artifact):
        """All required keys must be present in the artifact."""
        required_keys = [
            'pipeline', 'label_encoder', 'feature_names',
            'random_seed', 'cv_f1_macro', 'metrics', 'metadata'
        ]
        for key in required_keys:
            assert key in artifact, f"Missing key: '{key}' in model artifact"

    def test_metrics_dict_present(self, artifact):
        """Metrics dictionary must be present with required keys."""
        assert 'metrics' in artifact, "Missing 'metrics' dictionary"
        
        required_metrics = [
            'cv_f1_macro', 'cv_f1_std', 'cv_accuracy', 'cv_accuracy_std',
            'cv_precision_macro', 'cv_recall_macro', 'train_f1_macro',
            'train_f1_std', 'overfitting_gap'
        ]
        
        for metric in required_metrics:
            assert metric in artifact['metrics'], \
                f"Missing metric: '{metric}' in artifact['metrics']"

    def test_per_class_metrics_present(self, artifact):
        """Per-class metrics must be present for all classes."""
        assert 'per_class_metrics' in artifact, "Missing 'per_class_metrics'"
        
        per_class = artifact['per_class_metrics']
        for class_name in EXPECTED_CLASSES:
            assert class_name in per_class, \
                f"Missing per-class metrics for '{class_name}'"
            
            # Check required keys per class
            required_keys = ['precision', 'recall', 'f1-score', 'support']
            for key in required_keys:
                assert key in per_class[class_name], \
                    f"Missing '{key}' in per_class_metrics['{class_name}']"

    def test_correct_random_seed(self, artifact):
        """Model must be trained with seed 4."""
        assert artifact['random_seed'] == EXPECTED_SEED, \
            f"Expected seed {EXPECTED_SEED}, got {artifact['random_seed']}"

    def test_minimum_f1_macro(self, artifact):
        """CV F1-Macro must meet minimum threshold."""
        cv_f1 = artifact['cv_f1_macro']
        assert cv_f1 >= MIN_CV_F1_MACRO, \
            f"CV F1-Macro {cv_f1:.2%} is below minimum {MIN_CV_F1_MACRO:.2%}"

    def test_f1_macro_in_metrics(self, artifact):
        """F1-Macro must also exist in metrics dictionary."""
        cv_f1_top = artifact['cv_f1_macro']
        cv_f1_metrics = artifact['metrics']['cv_f1_macro']
        
        # Should be the same value
        assert abs(cv_f1_top - cv_f1_metrics) < 1e-6, \
            f"F1-Macro mismatch: top-level={cv_f1_top}, metrics={cv_f1_metrics}"

    def test_optimization_metric(self, artifact):
        """Metadata must indicate f1_macro as optimization metric."""
        opt_metric = artifact['metadata'].get('optimization_metric')
        assert opt_metric == 'f1_macro', \
            f"Expected optimization_metric='f1_macro', got '{opt_metric}'"

    def test_correct_classes(self, label_encoder):
        """Label encoder must have exactly 5 correct classes."""
        classes = list(label_encoder.classes_)
        assert classes == EXPECTED_CLASSES, \
            f"Expected classes {EXPECTED_CLASSES}, got {classes}"

    def test_correct_feature_names(self, feature_names):
        """Model must use exactly the 8 expected features."""
        assert list(feature_names) == EXPECTED_FEATURES, \
            f"Expected features {EXPECTED_FEATURES}, got {list(feature_names)}"

    def test_correct_feature_count(self, feature_names):
        """Model must have exactly 8 input features."""
        assert len(feature_names) == 8, \
            f"Expected 8 features, got {len(feature_names)}"

    def test_pipeline_has_correct_steps(self, pipeline):
        """Pipeline must contain poly, scaler, and rfecv steps."""
        step_names = [name for name, _ in pipeline.steps]
        assert 'poly' in step_names, "Pipeline missing 'poly' step"
        assert 'scaler' in step_names, "Pipeline missing 'scaler' step"
        assert 'rfecv' in step_names, "Pipeline missing 'rfecv' step"

    def test_selected_features_count(self, artifact):
        """RFECV must have selected at least 1 feature."""
        n_selected = artifact['metadata']['n_features_selected']
        assert n_selected >= 1, \
            f"Expected at least 1 selected feature, got {n_selected}"
        
        # Also check it's reasonable (not all 44 polynomial features)
        assert n_selected <= 44, \
            f"Too many selected features: {n_selected}"

    def test_overfitting_gap_reasonable(self, artifact):
        """Overfitting gap must be reasonable (not too high)."""
        gap = artifact['metrics']['overfitting_gap']
        assert gap < 20, \
            f"Overfitting gap too high: {gap:.2f}% (indicates overfitting)"

    def test_model_file_size(self):
        """Model file must be under 50MB."""
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        assert size_mb < 50, \
            f"Model file too large: {size_mb:.2f}MB (limit: 50MB)"

# ==========================================
# 2. INFERENCE TESTS
# ==========================================

class TestInference:
    """Validate that inference generates consistent and logically correct outputs."""

    def test_predict_returns_valid_class(self, pipeline, label_encoder, dummy_sample):
        """Prediction must return a valid face shape class."""
        pred_idx = pipeline.predict(dummy_sample)[0]
        shape = label_encoder.classes_[pred_idx]
        assert shape in EXPECTED_CLASSES, \
            f"Prediction '{shape}' is not a valid class"

    def test_predict_proba_sums_to_one(self, pipeline, dummy_sample):
        """Probability outputs must sum to 1.0."""
        proba = pipeline.predict_proba(dummy_sample)[0]
        assert abs(proba.sum() - 1.0) < 1e-6, \
            f"Probabilities sum to {proba.sum():.6f}, expected 1.0"

    def test_predict_proba_all_positive(self, pipeline, dummy_sample):
        """All class probabilities must be non-negative."""
        proba = pipeline.predict_proba(dummy_sample)[0]
        assert all(p >= 0 for p in proba), \
            f"Negative probability found: {proba}"

    def test_predict_proba_correct_shape(self, pipeline, dummy_sample):
        """Probability output must have exactly 5 values (one per class)."""
        proba = pipeline.predict_proba(dummy_sample)[0]
        assert len(proba) == 5, \
            f"Expected 5 class probabilities, got {len(proba)}"

    def test_confidence_is_reasonable(self, pipeline, dummy_sample):
        """Confidence (max probability) must be between 0 and 1."""
        proba = pipeline.predict_proba(dummy_sample)[0]
        confidence = np.max(proba)
        assert 0.0 <= confidence <= 1.0, \
            f"Confidence {confidence} is out of range [0, 1]"

    def test_inference_accepts_dataframe(self, pipeline, feature_names):
        """Pipeline must accept pandas DataFrame input."""
        df = pd.DataFrame([{
            'ratio_len_width': 1.20,
            'ratio_jaw_cheek': 0.85,
            'ratio_forehead_jaw': 0.90,
            'avg_jaw_angle': 140.0,
            'ratio_chin_jaw': 0.85,
            'circularity': 0.92,
            'solidity': 0.998,
            'extent': 0.78
        }], columns=feature_names)

        try:
            result = pipeline.predict(df)
            assert result is not None
        except Exception as e:
            pytest.fail(f"Pipeline failed on valid DataFrame input: {e}")

    def test_multiple_samples_inference(self, pipeline, label_encoder, feature_names):
        """Pipeline must handle multiple samples correctly."""
        samples = pd.DataFrame([
            {
                'ratio_len_width': 1.18, 'ratio_jaw_cheek': 0.88,
                'ratio_forehead_jaw': 0.85, 'avg_jaw_angle': 145.0,
                'ratio_chin_jaw': 0.88, 'circularity': 0.95,
                'solidity': 0.999, 'extent': 0.79
            },
            {
                'ratio_len_width': 1.05, 'ratio_jaw_cheek': 0.92,
                'ratio_forehead_jaw': 0.80, 'avg_jaw_angle': 130.0,
                'ratio_chin_jaw': 0.82, 'circularity': 0.98,
                'solidity': 0.997, 'extent': 0.82
            }
        ], columns=feature_names)

        preds = pipeline.predict(samples)
        assert len(preds) == 2, \
            f"Expected 2 predictions, got {len(preds)}"
        for pred in preds:
            assert label_encoder.classes_[pred] in EXPECTED_CLASSES

# ==========================================
# 3. INPUT VALIDATION TESTS
# ==========================================

class TestInputValidation:
    """Verify model robustness against boundary and extreme input values."""

    def test_boundary_ratios(self, pipeline, label_encoder, feature_names):
        """Model must handle boundary feature values without crashing."""
        boundary_sample = pd.DataFrame([{
            'ratio_len_width': 0.5,
            'ratio_jaw_cheek': 0.5,
            'ratio_forehead_jaw': 0.5,
            'avg_jaw_angle': 90.0,
            'ratio_chin_jaw': 0.5,
            'circularity': 0.5,
            'solidity': 0.9,
            'extent': 0.5
        }], columns=feature_names)

        try:
            pred = pipeline.predict(boundary_sample)[0]
            shape = label_encoder.classes_[pred]
            assert shape in EXPECTED_CLASSES
        except Exception as e:
            pytest.fail(f"Model failed on boundary values: {e}")

    def test_high_ratio_values(self, pipeline, label_encoder, feature_names):
        """Model must handle high feature values without crashing."""
        high_sample = pd.DataFrame([{
            'ratio_len_width': 2.0,
            'ratio_jaw_cheek': 1.2,
            'ratio_forehead_jaw': 1.5,
            'avg_jaw_angle': 170.0,
            'ratio_chin_jaw': 1.1,
            'circularity': 1.0,
            'solidity': 1.0,
            'extent': 0.95
        }], columns=feature_names)

        try:
            pred = pipeline.predict(high_sample)[0]
            shape = label_encoder.classes_[pred]
            assert shape in EXPECTED_CLASSES
        except Exception as e:
            pytest.fail(f"Model failed on high feature values: {e}")

# ==========================================
# 4. API CODE TESTS
# ==========================================

class TestAPICode:
    """Confirm that api.py includes mandatory endpoints and valid structure."""

    def test_api_file_exists(self):
        """api.py must exist at expected path."""
        assert os.path.exists(API_PATH), f"api.py not found at {API_PATH}"

    def test_api_syntax_valid(self):
        """api.py must have valid Python syntax."""
        import ast
        with open(API_PATH, 'r') as f:
            source = f.read()
        try:
            ast.parse(source)
        except SyntaxError as e:
            pytest.fail(f"api.py has syntax error: {e}")

    def test_api_required_functions(self):
        """api.py must contain all required endpoint functions."""
        import ast
        with open(API_PATH, 'r') as f:
            source = f.read()

        tree = ast.parse(source)
        functions = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        required = ['extract_features', 'analyze_skintone', 'home', 'predict_face']

        for func in required:
            assert func in functions, \
                f"Required function '{func}()' not found in api.py"

    def test_api_uses_pipeline(self):
        """api.py must use 'pipeline' (not old 'model' variable)."""
        with open(API_PATH, 'r') as f:
            source = f.read()
        assert "pipeline" in source, \
            "api.py must use 'pipeline' variable (new model structure)"
        assert "artifact['pipeline']" in source, \
            "api.py must load 'pipeline' from artifact"

    def test_api_correct_response_keys(self):
        """Ensure api.py returns all mandatory response fields required by the Android client."""
        with open(API_PATH, 'r') as f:
            source = f.read()
        required_keys = ['"status"', '"shape"', '"skintone"', '"server_inference_ms"']
        for key in required_keys:
            assert key in source, \
                f"api.py response missing key: {key}"

# ==========================================
# 5. METRICS VALIDATION TESTS (NEW)
# ==========================================

class TestMetricsValidation:
    """Validate that all metrics are within reasonable ranges."""

    def test_all_metrics_are_floats(self, artifact):
        """All metric values must be numeric."""
        metrics = artifact['metrics']
        for key, value in metrics.items():
            assert isinstance(value, (int, float, np.number)), \
                f"Metric '{key}' is not numeric: {type(value)}"

    def test_f1_scores_in_valid_range(self, artifact):
        """F1 scores must be between 0 and 1."""
        metrics = artifact['metrics']
        
        f1_keys = ['cv_f1_macro', 'train_f1_macro']
        for key in f1_keys:
            value = metrics[key]
            assert 0.0 <= value <= 1.0, \
                f"{key} = {value} is out of valid range [0, 1]"

    def test_std_values_reasonable(self, artifact):
        """Standard deviation values must be positive and reasonable."""
        metrics = artifact['metrics']
        
        std_keys = ['cv_f1_std', 'train_f1_std', 'cv_accuracy_std']
        for key in std_keys:
            value = metrics[key]
            assert 0.0 <= value <= 0.5, \
                f"{key} = {value} is unreasonably high (> 0.5)"

    def test_per_class_f1_scores_valid(self, artifact):
        """Per-class F1 scores must be between 0 and 1."""
        per_class = artifact['per_class_metrics']
        
        for class_name, metrics in per_class.items():
            f1 = metrics['f1-score']
            assert 0.0 <= f1 <= 1.0, \
                f"F1 for '{class_name}' = {f1} is out of range [0, 1]"

    def test_support_values_correct(self, artifact):
        """Support values must be positive integers summing to total samples."""
        per_class = artifact['per_class_metrics']
        
        total_support = sum(m['support'] for m in per_class.values())
        expected_support = artifact['metadata']['n_samples']
        
        assert total_support == expected_support, \
            f"Total support {total_support} != expected {expected_support}"

    def test_train_better_than_cv(self, artifact):
        """Training F1 should typically be >= CV F1 (can be equal for good models)."""
        metrics = artifact['metrics']
        train_f1 = metrics['train_f1_macro']
        cv_f1 = metrics['cv_f1_macro']
        
        # Allow small margin for statistical variation
        assert train_f1 >= cv_f1 - 0.05, \
            f"Train F1 ({train_f1:.4f}) suspiciously lower than CV F1 ({cv_f1:.4f})"
        