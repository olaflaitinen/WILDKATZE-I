import hashlib
import hmac
import secrets
from typing import Optional

def generate_api_key() -> str:
    """Generate a secure API key."""
    return secrets.token_urlsafe(32)

def hash_api_key(api_key: str) -> str:
    """Hash an API key for storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()

def verify_api_key(provided_key: str, stored_hash: str) -> bool:
    """Verify an API key against stored hash."""
    provided_hash = hash_api_key(provided_key)
    return hmac.compare_digest(provided_hash, stored_hash)

def sanitize_input(text: str, max_length: int = 32768) -> str:
    """Sanitize user input."""
    if not text:
        return ""
    # Truncate
    text = text[:max_length]
    # Remove null bytes
    text = text.replace('\x00', '')
    return text

def redact_pii(text: str) -> str:
    """Redact personally identifiable information."""
    import re
    # Email
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL REDACTED]', text)
    # Phone
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE REDACTED]', text)
    # SSN-like
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN REDACTED]', text)
    return text

class AuditLogger:
    """Logs security-relevant events."""
    
    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path
        
    def log_event(self, event_type: str, user_id: str, details: dict):
        import json
        from datetime import datetime
        
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "details": details
        }
        
        if self.log_path:
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(event) + "\n")
