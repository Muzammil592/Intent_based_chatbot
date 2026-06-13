import os
import random
import pickle
import sys
import re

class SessionContext:
    """Tracks conversation history states across turns."""
    def __init__(self):
        self.last_intent = None
        self.extracted_entities = {}

    def reset(self):
        self.last_intent = None
        self.extracted_entities = {}

def extract_entities(text: str) -> dict:
    """Pure regex entity extraction matrix for locations and times."""
    entities = {}
    
    # 1. Look for time parameters (e.g., 5pm, 10:30 am, 7am)
    time_match = re.search(r'\b(\d{1,2}(?::\d{2})?\s*(?:am|pm|am|pm))\b', text, re.IGNORECASE)
    if time_match:
        entities['time'] = time_match.group(1)
        
    # 2. Look for explicit typical Pakistani cities/locations as target parameters
    location_match = re.search(r'\b(lahore|karachi|islamabad|uet|hostel|office|home)\b', text, re.IGNORECASE)
    if location_match:
        entities['location'] = location_match.group(1).capitalize()
        
    return entities

def live_chat_loop():
    print("\n--- Day 2: Advanced NLP Intent & Entity Engine ---")
    
    try:
        with open('/tmp/artifacts/preprocessor.pkl', 'rb') as f: preprocessor = pickle.load(f)
        with open('/tmp/artifacts/model.pkl', 'rb') as f: model = pickle.load(f)
        with open('/tmp/artifacts/responses.pkl', 'rb') as f: responses = pickle.load(f)
    except FileNotFoundError:
        print("[CRITICAL] Trained model assets missing! Build Docker image or run train.py.")
        sys.exit(1)

    context = SessionContext()
    print("Engine Status: ACTIVE. Threshold set to 0.45. Context memory tracking initialized.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower().strip() == 'exit':
            break
            
        if not user_input.strip():
            continue

        # Extract entities straight from raw input string before text processing clears it
        found_entities = extract_entities(user_input)
        context.extracted_entities.update(found_entities)

        # Vector space classification
        query_vec = preprocessor.transform_query(user_input)
        tag, confidence = model.predict_intent(query_vec)

        # UPGRADED: Day 2 strict confidence threshold handling
        if confidence < 0.45:
            print(f"Bot: [Confidence Matrix Low ({confidence:.2f})] I am not certain about your intent. Could you rephrase?")
            context.last_intent = "unknown"
            continue

        # Process contextual entity injection overrides
        if tag == "reminder" and 'time' in context.extracted_entities:
            reply = f"Understood. I have logged an automatic system alert for you exactly at {context.extracted_entities['time']}."
        elif tag == "weather" and 'location' in context.extracted_entities:
            reply = f"Checking real-time atmospheric updates for {context.extracted_entities['location']}... It is currently sunny and clear."
        elif tag == "schedule" and context.last_intent == "greet":
            reply = "Great to see you again! Checking your timeline records... You have a design review setup today."
        else:
            reply = random.choice(responses[tag])

        # Track contextual conversation memory
        context.last_intent = tag
        
        print(f"Bot (Intent: {tag} | Conf: {confidence:.2f} | Entities: {context.extracted_entities}): {reply}")

if __name__ == "__main__":
    live_chat_loop()