import re
from typing import List, Optional

class TextPreprocessor:
    """
    Standard preprocessing pipeline for military text data.
    """
    def __init__(self, strip_html: bool = True, lowercase: bool = False, remove_special: bool = False):
        self.strip_html = strip_html
        self.lowercase = lowercase
        self.remove_special = remove_special

    def clean(self, text: str) -> str:
        if not text:
            return ""
            
        cleaned = text
        
        if self.strip_html:
            cleaned = re.sub(r'<[^>]+>', '', cleaned)
            
        if self.lowercase:
            cleaned = cleaned.lower()
            
        if self.remove_special:
            cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', cleaned)
            
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned

    def process_batch(self, texts: List[str]) -> List[str]:
        return [self.clean(t) for t in texts]

def load_and_preprocess(filepath: str) -> List[str]:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = f.readlines()
    
    processor = TextPreprocessor()
    return processor.process_batch(data)
