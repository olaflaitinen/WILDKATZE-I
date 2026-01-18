import pytest

def test_input_sanitization():
    """Test input sanitization."""
    from wildkatze.utils.security import sanitize_input
    
    # Test truncation
    long_text = "a" * 50000
    result = sanitize_input(long_text, max_length=1000)
    assert len(result) == 1000
    
    # Test null byte removal
    text_with_null = "hello\x00world"
    result = sanitize_input(text_with_null)
    assert "\x00" not in result

def test_pii_redaction():
    """Test PII redaction."""
    from wildkatze.utils.security import redact_pii
    
    text = "Contact me at test@example.com or 123-456-7890"
    result = redact_pii(text)
    
    assert "test@example.com" not in result
    assert "123-456-7890" not in result

def test_api_key_generation():
    """Test API key generation."""
    from wildkatze.utils.security import generate_api_key, hash_api_key, verify_api_key
    
    key = generate_api_key()
    assert len(key) > 20
    
    hashed = hash_api_key(key)
    assert verify_api_key(key, hashed)
    assert not verify_api_key("wrong_key", hashed)
