import os
import random
import yaml
import logging
import joblib
import re
from flask import Flask, request, jsonify

# Initialize Core Microservice Components
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load Immutable Architecture Config Targets
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)
    
ARTIFACTS_DIR = config['system']['artifacts_dir']
THRESHOLD = config['model']['confidence_threshold']

# Volatile In-Memory Session Storage Context Matrix
session_context = {"last_intent": None, "extracted_entities": {}}

try:
    preprocessor = joblib.load(f"{ARTIFACTS_DIR}/preprocessor.joblib")
    model = joblib.load(f"{ARTIFACTS_DIR}/model.joblib")
    responses = joblib.load(f"{ARTIFACTS_DIR}/responses.joblib")
    logger.info("Production compiled weights securely mounted into memory layers.")
except FileNotFoundError:
    logger.warning("Artifact matrices missing from disk. Launching emergency fallback compilation.")
    import train
    train.execute_production_train()
    preprocessor = joblib.load(f"{ARTIFACTS_DIR}/preprocessor.joblib")
    model = joblib.load(f"{ARTIFACTS_DIR}/model.joblib")
    responses = joblib.load(f"{ARTIFACTS_DIR}/responses.joblib")

def run_entity_extraction(text: str) -> dict:
    entities = {}
    time_match = re.search(r'\b(\d{1,2}(?::\d{2})?\s*(?:am|pm))\b', text, re.IGNORECASE)
    if time_match:
        entities['time'] = time_match.group(1).lower()
    location_match = re.search(r'\b(lahore|karachi|islamabad|uet|office|home)\b', text, re.IGNORECASE)
    if location_match:
        entities['location'] = location_match.group(1).capitalize()
    return entities

@app.route('/chat', methods=['POST'])
def process_chat_intent():
    try:
        data = request.get_json()
        
        # Guardrail Input Layer: Request Field Validation
        if not data or 'message' not in data:
            logger.warning("API endpoint received an invalid, parameterless input block.")
            return jsonify({"error": "Malformed payload mapping. 'message' string element required."}), 400

        raw_message = str(data['message']).strip()
        
        # Guardrail Input Layer: Input Sanitization 
        if not raw_message or len(raw_message) > 500:
            return jsonify({"error": "Message structure failed sizing constraints (1-500 chars)."}), 400

        # Perform Processing Actions
        found_entities = run_entity_extraction(raw_message)
        session_context['extracted_entities'].update(found_entities)

        query_vec = preprocessor.transform_query(raw_message)
        tag, confidence = model.predict_intent(query_vec)

        # Handle Confidence Fallback Constraints
        if confidence < THRESHOLD:
            logger.info(f"Query dropped to fallback. Intent: {tag} returned inadequate confidence score: {confidence:.2f}")
            return jsonify({
                "intent": "unknown",
                "confidence": float(confidence),
                "entities": session_context['extracted_entities'],
                "response": "I am unable to clearly verify your request parameter intent. Please restructure your wording."
            }), 200

        # Process Response Injection Overrides
        if tag == "reminder" and 'time' in session_context['extracted_entities']:
            reply = f"Understood. I have logged an automatic system alert for you exactly at {session_context['extracted_entities']['time']}."
        elif tag == "weather" and 'location' in session_context['extracted_entities']:
            reply = f"Checking real-time atmospheric updates for {session_context['extracted_entities']['location']}... Sunny skies detected."
        else:
            reply = random.choice(responses[tag])

        session_context['last_intent'] = tag
        logger.info(f"Successfully processed response. Intent Match: '{tag}' (Conf: {confidence:.2f})")

        return jsonify({
            "intent": tag,
            "confidence": float(confidence),
            "entities": session_context['extracted_entities'],
            "response": reply
        }), 200

    except Exception as e:
        logger.error(f"Critical runtime exception encountered during inference mapping: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal System Error occurred inside the NLP engine."}), 500

if __name__ == "__main__":
    app.run(host=config['api']['host'], port=config['api']['port'], debug=config['api']['debug'])