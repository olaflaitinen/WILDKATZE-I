from typing import Dict, List, Tuple
import re
import logging

logger = logging.getLogger(__name__)

PROTECTED_ATTRIBUTES = ["race", "gender", "religion", "nationality", "ethnicity", "age", "disability"]

BIAS_PATTERNS = {
    "gender": [
        r"\b(all (?:men|women) are)\b",
        r"\b((?:men|women) can't)\b",
    ],
    "race": [
        r"\b(all (?:black|white|asian) people)\b",
    ],
    "religion": [
        r"\b(all (?:muslims|christians|jews|hindus) are)\b",
    ],
}

class BiasDetector:
    """Detects and scores bias in generated content."""
    
    def __init__(self):
        self.patterns = BIAS_PATTERNS
        
    def detect(self, text: str) -> Dict[str, List[str]]:
        """Detect bias patterns in text."""
        detected = {}
        text_lower = text.lower()
        
        for category, patterns in self.patterns.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, text_lower, re.IGNORECASE)
                matches.extend(found)
            if matches:
                detected[category] = matches
                
        return detected
    
    def score(self, text: str) -> float:
        """Return bias score (0 = no bias, 1 = highly biased)."""
        detected = self.detect(text)
        if not detected:
            return 0.0
            
        # Score based on number of categories and matches
        total_matches = sum(len(v) for v in detected.values())
        category_count = len(detected)
        
        # Normalize to 0-1 range
        score = min(1.0, (total_matches * 0.1) + (category_count * 0.2))
        return score
    
    def get_report(self, text: str) -> Dict:
        """Generate a bias detection report."""
        detected = self.detect(text)
        score = self.score(text)
        
        return {
            "score": score,
            "detected_patterns": detected,
            "categories_flagged": list(detected.keys()),
            "compliant": score < 0.05,
            "recommendation": "Review and revise" if score >= 0.05 else "No action needed"
        }
