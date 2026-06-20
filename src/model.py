import yaml
from sklearn.linear_model import LogisticRegression

class IntentClassifier:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)['model']
            
        self.classifier = LogisticRegression(
            C=config['logistic_regression_c'], 
            max_iter=1000, 
            class_weight='balanced', 
            random_state=42
        )

    def train_model(self, feature_matrix, labels: list):
        self.classifier.fit(feature_matrix, labels)

    def predict_intent(self, query_vector):
        prediction = self.classifier.predict(query_vector)[0]
        probabilities = self.classifier.predict_proba(query_vector)[0]
        max_confidence = max(probabilities)
        return prediction, max_confidence