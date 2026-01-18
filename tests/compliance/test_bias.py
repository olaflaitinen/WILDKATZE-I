import pytest

def test_bias_detector():
    """Test bias detection functionality."""
    from wildkatze.evaluation.bias_detection import BiasDetector
    
    detector = BiasDetector()
    
    # Test neutral text
    score = detector.score("The weather is nice today.")
    assert score < 0.1
    
    # Test biased text
    biased_report = detector.get_report("All members of group X are bad.")
    assert biased_report["score"] > 0.5

def test_protected_attributes():
    """Test detection of bias against protected attributes."""
    from wildkatze.evaluation.bias_detection import PROTECTED_ATTRIBUTES
    
    expected = ["race", "gender", "religion", "nationality", "ethnicity", "age", "disability"]
    for attr in expected:
        assert attr in PROTECTED_ATTRIBUTES
