import sys
import os

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_preprocessing_import():
    """Test that preprocessing.py can be imported."""
    try:
        import preprocessing
        assert True
    except ImportError as e:
        assert False, f"Failed to import preprocessing.py: {e}"

def test_tuning_import():
    """Test that hyperparameter_tuning.py can be imported."""
    try:
        import hyperparameter_tuning
        assert True
    except ImportError as e:
        assert False, f"Failed to import hyperparameter_tuning.py: {e}"
