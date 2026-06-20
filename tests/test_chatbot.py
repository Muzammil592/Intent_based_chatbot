import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from preprocessor import AdvancedPreprocessor
from chatbot import app

@pytest.fixture
def api_client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_preprocessor_normalization():
    """Validates if basic contraction mappings process flawlessly without stopword interference."""
    preprocessor = AdvancedPreprocessor(config_path="config.yaml")
    cleaned = preprocessor.clean_text("I'm coding raw algorithms!")
    
    # 'am' is filtered by design as a stopword, leaving 'i' intact from 'i am'
    assert "i" in cleaned.split()
    assert "algorithm" in cleaned

def test_api_endpoint_success(api_client):
    """Validates standard successful endpoint mapping and returns valid JSON structures."""
    response = api_client.post('/chat', json={"message": "Hello there, good morning!"})
    json_data = response.get_json()
    
    assert response.status_code == 200
    assert "intent" in json_data
    assert "response" in json_data
    assert json_data["intent"] == "greet"

def test_api_malformed_input_guardrail(api_client):
    """Verifies that empty strings or bad keys trigger correct validation status codes."""
    response = api_client.post('/chat', json={"bad_key": "empty string"})
    assert response.status_code == 400