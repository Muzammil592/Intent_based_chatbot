import json
import os
import yaml
import logging
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from preprocessor import AdvancedPreprocessor
from model import IntentClassifier

# Initialize systems telemetry log layers
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

def execute_production_train():
    with open("config.yaml", 'r') as f:
        sys_config = yaml.safe_load(f)['system']
    
    logger.info("Initializing model training pipeline orchestration sequence.")
    data_path = os.path.join(os.path.dirname(__file__), '../data/intents.json')
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    patterns, labels, responses = [], [], {}
    for intent in data['intents']:
        tag = intent['tag']
        responses[tag] = intent['responses']
        for pattern in intent['patterns']:
            patterns.append(pattern)
            labels.append(tag)

    patterns, labels = np.array(patterns), np.array(labels)
    X_train_raw, _, y_train, _ = train_test_split(
        patterns, labels, test_size=0.20, random_state=42, stratify=labels
    )

    preprocessor = AdvancedPreprocessor()
    X_train_vec = preprocessor.fit_transform_matrix(X_train_raw.tolist())

    model = IntentClassifier()
    model.train_model(X_train_vec, y_train.tolist())
    logger.info(f"Convex optimization complete. Shape weights: {X_train_vec.shape}")

    # Persist model data frames securely via Joblib
    art_dir = sys_config['artifacts_dir']
    os.makedirs(art_dir, exist_ok=True)
    
    joblib.dump(preprocessor, f"{art_dir}/preprocessor.joblib")
    joblib.dump(model, f"{art_dir}/model.joblib")
    joblib.dump(responses, f"{art_dir}/responses.joblib")
    logger.info(f"Production pipeline artifacts successfully written to disk at: {art_dir}")

if __name__ == "__main__":
    execute_production_train()