import json
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Explicitly import the correct upgraded class from your preprocessor module
from preprocessor import AdvancedPreprocessor
from model import IntentClassifier

def run_evaluation_and_training():
    data_path = os.path.join(os.path.dirname(__file__), '../data/intents.json')
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    patterns = []
    labels = []
    responses = {}

    for intent in data['intents']:
        tag = intent['tag']
        responses[tag] = intent['responses']
        for pattern in intent['patterns']:
            patterns.append(pattern)
            labels.append(tag)

    # Convert to stable NumPy arrays
    patterns = np.array(patterns)
    labels = np.array(labels)

    # 80/20 Train/Test Validation Split Matrix
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        patterns, labels, test_size=0.20, random_state=42, stratify=labels
    )

    print(f"Total Patterns: {len(patterns)} | Training Subset: {len(X_train_raw)} | Test Validation Subset: {len(X_test_raw)}")

    # FIXED: Calling AdvancedPreprocessor cleanly
    preprocessor = AdvancedPreprocessor()
    X_train_vec = preprocessor.fit_transform_matrix(X_train_raw.tolist())
    X_test_vec = preprocessor.vectorizer.transform([preprocessor.clean_text(text) for text in X_test_raw.tolist()])

    # Train structural classifier weights
    model = IntentClassifier()
    model.train_model(X_train_vec, y_train.tolist())

    # Evaluation calculations output layer
    print("\n================ DAY 2 CLASSIFICATION PERFORMANCE ================")
    y_pred = [model.classifier.predict(vec)[0] for vec in X_test_vec]
    print(classification_report(y_test, y_pred, zero_division=0))
    
    print("======================= CONFUSION MATRIX =======================")
    print(confusion_matrix(y_test, y_pred))
    print("================================================================\n")

    # Serialize system artifacts securely using Pickle
    os.makedirs('/tmp/artifacts', exist_ok=True)
    with open('/tmp/artifacts/preprocessor.pkl', 'wb') as f: pickle.dump(preprocessor, f)
    with open('/tmp/artifacts/model.pkl', 'wb') as f: pickle.dump(model, f)
    with open('/tmp/artifacts/responses.pkl', 'wb') as f: pickle.dump(responses, f)
    print("All optimization artifacts successfully committed.")

if __name__ == "__main__":
    # FIXED: Aligning structural call to the exact function name above
    run_evaluation_and_training()