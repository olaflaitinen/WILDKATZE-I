"""
WILDKATZE-I Cultural Sensitivity Analysis

This module implements cultural sensitivity analysis for WILDKATZE-I,
including:
- Cultural appropriateness scoring
- Cultural adaptation of messages
- Cross-cultural communication analysis
- Taboo and sensitivity detection

The analyzer uses a comprehensive cultural database covering 100+ cultures
with dimensions based on Hofstede's cultural theory and regional expertise.

Copyright (c) 2026 olaflaitinen. All rights reserved.
Licensed under EUPL v1.2
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class CulturalProfile:
    """
    Profile representing a culture's characteristics.
    
    Attributes:
        code: ISO culture code (e.g., "de-DE")
        name: Human-readable culture name
        values: List of core cultural values
        taboos: List of cultural taboos
        communication_style: Communication context (high/low/medium)
        religious_considerations: Religious factors
        greeting_conventions: Greeting protocols
        formality_level: Expected formality (1-10)
    """
    code: str
    name: str
    values: List[str]
    taboos: List[str]
    communication_style: str
    religious_considerations: List[str] = None
    greeting_conventions: List[str] = None
    formality_level: int = 5
    
    def __post_init__(self):
        if self.religious_considerations is None:
            self.religious_considerations = []
        if self.greeting_conventions is None:
            self.greeting_conventions = []


@dataclass
class SensitivityIssue:
    """
    Represents a detected sensitivity issue.
    
    Attributes:
        category: Type of issue (taboo, formality, religious, etc.)
        severity: Severity level (1-10)
        description: Human-readable description
        location: Position in text where issue was found
        suggestion: Suggested correction
    """
    category: str
    severity: int
    description: str
    location: Optional[Tuple[int, int]] = None
    suggestion: Optional[str] = None


class CulturalDatabase:
    """
    Database of cultural profiles and sensitivity rules.
    
    This class manages the cultural knowledge base used for
    cultural sensitivity analysis and adaptation.
    
    Args:
        database_path: Path to cultural contexts JSON file
    """
    
    def __init__(self, database_path: Optional[str] = None):
        self.profiles: Dict[str, CulturalProfile] = {}
        self.sensitivity_rules: Dict[str, List[Dict]] = {}
        
        if database_path:
            self.load_database(database_path)
        else:
            self._load_default_profiles()
            
    def _load_default_profiles(self):
        """Load default cultural profiles."""
        default_profiles = [
            CulturalProfile(
                code="de-DE",
                name="German",
                values=["punctuality", "efficiency", "directness", "privacy", "order"],
                taboos=["nazi_references", "informal_address_to_strangers", "loud_behavior"],
                communication_style="low_context",
                formality_level=7,
            ),
            CulturalProfile(
                code="ar-SA",
                name="Saudi Arabian",
                values=["hospitality", "family", "honor", "religion", "respect_for_elders"],
                taboos=["left_hand_gestures", "alcohol_references", "pork", "public_criticism"],
                communication_style="high_context",
                religious_considerations=["islamic_practices", "prayer_times", "ramadan"],
                formality_level=8,
            ),
            CulturalProfile(
                code="ps-AF",
                name="Pashto (Afghanistan)",
                values=["hospitality", "honor", "tribal_loyalty", "bravery", "pashtunwali"],
                taboos=["insulting_elders", "public_shame", "disrespecting_women"],
                communication_style="high_context",
                religious_considerations=["islamic_practices"],
                formality_level=9,
            ),
            CulturalProfile(
                code="zh-CN",
                name="Chinese (Simplified)",
                values=["harmony", "face", "hierarchy", "collectivism", "education"],
                taboos=["direct_confrontation", "number_4", "losing_face"],
                communication_style="high_context",
                formality_level=7,
            ),
            CulturalProfile(
                code="ru-RU",
                name="Russian",
                values=["strength", "resilience", "community", "tradition", "patriotism"],
                taboos=["weakness_display", "empty_gestures", "excessive_smiling"],
                communication_style="medium_context",
                formality_level=6,
            ),
            CulturalProfile(
                code="en-US",
                name="American English",
                values=["individualism", "freedom", "equality", "directness", "optimism"],
                taboos=["racism", "political_extremism"],
                communication_style="low_context",
                formality_level=4,
            ),
            CulturalProfile(
                code="ja-JP",
                name="Japanese",
                values=["harmony", "respect", "group_loyalty", "politeness", "perfectionism"],
                taboos=["direct_refusal", "public_disagreement", "pointing"],
                communication_style="high_context",
                formality_level=9,
            ),
        ]
        
        for profile in default_profiles:
            self.profiles[profile.code] = profile
            
    def load_database(self, path: str):
        """
        Load cultural database from JSON file.
        
        Args:
            path: Path to JSON file
        """
        path = Path(path)
        if not path.exists():
            logger.warning(f"Cultural database not found at {path}, using defaults")
            self._load_default_profiles()
            return
            
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        for culture_data in data.get("cultures", []):
            profile = CulturalProfile(
                code=culture_data["code"],
                name=culture_data["name"],
                values=culture_data.get("values", []),
                taboos=culture_data.get("taboos", []),
                communication_style=culture_data.get("communication_style", "medium_context"),
                religious_considerations=culture_data.get("religious_considerations", []),
                formality_level=culture_data.get("formality_level", 5),
            )
            self.profiles[profile.code] = profile
            
        logger.info(f"Loaded {len(self.profiles)} cultural profiles")
        
    def get_profile(self, culture_code: str) -> Optional[CulturalProfile]:
        """
        Get cultural profile by code.
        
        Args:
            culture_code: ISO culture code
            
        Returns:
            CulturalProfile or None if not found
        """
        return self.profiles.get(culture_code)
    
    def list_cultures(self) -> List[str]:
        """List all available culture codes."""
        return list(self.profiles.keys())


class CulturalSensitivityAnalyzer:
    """
    Analyzer for cultural sensitivity and appropriateness.
    
    This class provides methods to evaluate text for cultural
    appropriateness, identify potential issues, and suggest
    adaptations for specific target cultures.
    
    Args:
        cultural_database_path: Optional path to cultural database JSON
    """
    
    def __init__(self, cultural_database_path: Optional[str] = None):
        self.database = CulturalDatabase(cultural_database_path)
        self._compile_patterns()
        
    def _compile_patterns(self):
        """Compile regex patterns for sensitivity detection."""
        self.taboo_patterns = {
            "alcohol_references": [
                r"\b(beer|wine|alcohol|drunk|drinking|bar|pub)\b",
            ],
            "pork": [
                r"\b(pork|bacon|ham|pig)\b",
            ],
            "nazi_references": [
                r"\b(nazi|hitler|third reich|swastika)\b",
            ],
            "religious_sensitivity": [
                r"\b(infidel|crusade|jihad)\b",
            ],
            "political_extremism": [
                r"\b(terrorist|extremist|radical)\b",
            ],
        }
        
        self.formality_indicators = {
            "informal": [
                r"\b(gonna|wanna|gotta|ya|hey|hi)\b",
                r"!{2,}",
                r"\.{3,}",
            ],
            "formal": [
                r"\b(respectfully|hereby|therefore|pursuant)\b",
            ],
        }
        
    def evaluate(
        self, 
        text: str, 
        target_culture: str
    ) -> Tuple[float, List[SensitivityIssue]]:
        """
        Evaluate text for cultural appropriateness.
        
        Args:
            text: Text to analyze
            target_culture: Target culture code
            
        Returns:
            Tuple of (score 0-10, list of issues)
        """
        profile = self.database.get_profile(target_culture)
        if profile is None:
            logger.warning(f"Unknown culture: {target_culture}")
            return 5.0, [SensitivityIssue(
                category="unknown_culture",
                severity=5,
                description=f"Culture '{target_culture}' not in database"
            )]
            
        issues = []
        base_score = 10.0
        
        # Check for taboos
        taboo_issues = self._check_taboos(text, profile)
        issues.extend(taboo_issues)
        
        # Check formality
        formality_issues = self._check_formality(text, profile)
        issues.extend(formality_issues)
        
        # Check communication style
        style_issues = self._check_communication_style(text, profile)
        issues.extend(style_issues)
        
        # Check religious considerations
        religious_issues = self._check_religious_sensitivity(text, profile)
        issues.extend(religious_issues)
        
        # Calculate final score
        for issue in issues:
            penalty = issue.severity * 0.3
            base_score = max(0, base_score - penalty)
            
        return round(base_score, 1), issues
    
    def _check_taboos(
        self, 
        text: str, 
        profile: CulturalProfile
    ) -> List[SensitivityIssue]:
        """Check text for cultural taboos."""
        issues = []
        text_lower = text.lower()
        
        for taboo in profile.taboos:
            if taboo in self.taboo_patterns:
                for pattern in self.taboo_patterns[taboo]:
                    matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                    for match in matches:
                        issues.append(SensitivityIssue(
                            category="taboo",
                            severity=8,
                            description=f"Taboo content detected: {taboo}",
                            location=(match.start(), match.end()),
                            suggestion=f"Remove or replace reference to {taboo}"
                        ))
                        
        return issues
    
    def _check_formality(
        self, 
        text: str, 
        profile: CulturalProfile
    ) -> List[SensitivityIssue]:
        """Check text formality against cultural expectations."""
        issues = []
        
        informal_count = 0
        formal_count = 0
        
        for pattern in self.formality_indicators["informal"]:
            informal_count += len(re.findall(pattern, text, re.IGNORECASE))
            
        for pattern in self.formality_indicators["formal"]:
            formal_count += len(re.findall(pattern, text, re.IGNORECASE))
            
        expected_formality = profile.formality_level
        
        if expected_formality >= 7 and informal_count > 2:
            issues.append(SensitivityIssue(
                category="formality",
                severity=5,
                description="Text may be too informal for this culture",
                suggestion="Use more formal language and address forms"
            ))
            
        if expected_formality <= 3 and formal_count > 3:
            issues.append(SensitivityIssue(
                category="formality",
                severity=3,
                description="Text may be too formal for this culture",
                suggestion="Consider more casual language"
            ))
            
        return issues
    
    def _check_communication_style(
        self, 
        text: str, 
        profile: CulturalProfile
    ) -> List[SensitivityIssue]:
        """Check text against communication style expectations."""
        issues = []
        
        # High-context cultures prefer indirect communication
        if profile.communication_style == "high_context":
            direct_patterns = [
                r"\b(you must|you should|you need to)\b",
                r"\b(this is wrong|that is incorrect)\b",
            ]
            
            for pattern in direct_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    issues.append(SensitivityIssue(
                        category="communication_style",
                        severity=4,
                        description="Direct language may be inappropriate",
                        suggestion="Consider more indirect phrasing"
                    ))
                    break
                    
        return issues
    
    def _check_religious_sensitivity(
        self, 
        text: str, 
        profile: CulturalProfile
    ) -> List[SensitivityIssue]:
        """Check for religious sensitivity issues."""
        issues = []
        
        if "islamic_practices" in profile.religious_considerations:
            if re.search(r"\b(pork|alcohol|wine|beer)\b", text, re.IGNORECASE):
                issues.append(SensitivityIssue(
                    category="religious",
                    severity=7,
                    description="Content may conflict with Islamic practices",
                    suggestion="Remove references to prohibited items"
                ))
                
        return issues
    
    def adapt(
        self, 
        text: str, 
        source_culture: str, 
        target_culture: str
    ) -> str:
        """
        Adapt text from one culture to another.
        
        Args:
            text: Original text
            source_culture: Source culture code
            target_culture: Target culture code
            
        Returns:
            Culturally adapted text
        """
        target_profile = self.database.get_profile(target_culture)
        if target_profile is None:
            return text
            
        adapted = text
        
        # Formality adaptation
        if target_profile.formality_level >= 7:
            adapted = self._increase_formality(adapted)
        elif target_profile.formality_level <= 3:
            adapted = self._decrease_formality(adapted)
            
        # Remove taboo content
        for taboo in target_profile.taboos:
            if taboo in self.taboo_patterns:
                for pattern in self.taboo_patterns[taboo]:
                    adapted = re.sub(pattern, "[adapted content]", adapted, flags=re.IGNORECASE)
                    
        return adapted
    
    def _increase_formality(self, text: str) -> str:
        """Increase text formality."""
        replacements = {
            r"\bhi\b": "Greetings",
            r"\bhey\b": "Greetings",
            r"\bgonna\b": "going to",
            r"\bwanna\b": "want to",
            r"\bgotta\b": "have to",
        }
        
        result = text
        for pattern, replacement in replacements.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
            
        return result
    
    def _decrease_formality(self, text: str) -> str:
        """Decrease text formality."""
        replacements = {
            r"\bGood morning\b": "Hi",
            r"\bGood afternoon\b": "Hi",
            r"\bI would like to\b": "I wanna",
        }
        
        result = text
        for pattern, replacement in replacements.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
            
        return result
    
    def get_cultural_summary(self, culture_code: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of cultural characteristics.
        
        Args:
            culture_code: ISO culture code
            
        Returns:
            Dictionary with cultural summary or None
        """
        profile = self.database.get_profile(culture_code)
        if profile is None:
            return None
            
        return {
            "code": profile.code,
            "name": profile.name,
            "values": profile.values,
            "communication_style": profile.communication_style,
            "formality_level": profile.formality_level,
            "key_considerations": profile.taboos[:3] + profile.religious_considerations[:2],
        }
