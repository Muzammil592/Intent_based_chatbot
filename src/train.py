import json
import os
import pickle
from preprocessor import TextPreprocessor
from model import IntentClassifier

def run_pipeline():
    data_path = os.path.join(os.path.dirname(__file__), '../data/intents.json')
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    patterns = []
    labels = []
    responses = {}

    # Extract pattern tokens and map them to their classification classes
    for intent in data['intents']:
        tag = intent['tag']
        responses[tag] = intent['responses']
        for pattern in intent['patterns']:
            patterns.append(pattern)
            labels.append(tag)

    # Execute Preprocessing Feature Vectors
    preprocessor = TextPreprocessor()
    feature_matrix = preprocessor.fit_transform_matrix(patterns)

    # Train Classifier
    model = IntentClassifier()
    model.train_model(feature_matrix, labels)
    print(f"-> Successfully trained weights on {feature_matrix.shape[0]} structural patterns.")

    # Serialize artifacts directly using Pickle to avoid re-training steps during container boot
    os.makedirs('/tmp/artifacts', exist_ok=True)
    with open('/tmp/artifacts/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    with open('/tmp/artifacts/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('/tmp/artifacts/responses.pkl', 'wb') as f:
        pickle.dump(responses, f)
    print("-> System artifacts securely saved.")

if __name__ == "__main__":
    run_pipeline()