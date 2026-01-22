"""
Tests for data preprocessing module.
"""

import pytest
import pandas as pd
from src.data.preprocessor import DataPreprocessor


@pytest.fixture
def preprocessor():
    """Create DataPreprocessor instance."""
    return DataPreprocessor()


def test_categorize_age(preprocessor):
    """Test age categorization."""
    assert preprocessor.categorize_age(15) == "Teen"
    assert preprocessor.categorize_age(25) == "Adult"
    assert preprocessor.categorize_age(65) == "Old_Age"
    assert preprocessor.categorize_age(None) == "Unknown"


def test_categorize_weapon(preprocessor):
    """Test weapon categorization."""
    assert preprocessor.categorize_weapon("Handgun") == "Handgun"
    assert preprocessor.categorize_weapon("Multiple Handguns") == "Multiple_Weapons"
    assert preprocessor.categorize_weapon(None) == "Unknown/Not_Specified"


def test_categorize_situation(preprocessor):
    """Test situation categorization."""
    assert preprocessor.categorize_situation("Bullying") == "Conflict_Related"
    assert preprocessor.categorize_situation("Illegal Activity") == "Criminal_Activity"
    assert preprocessor.categorize_situation("Accidental") == "Accidental"
    assert preprocessor.categorize_situation(None) == "Unknown"
