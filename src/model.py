from sklearn.linear_model import LogisticRegression

class IntentClassifier:
    def __init__(self):
        # Increased C parameter (Lower regularization) allows strict fitting on short-text tokens
        # class_weight='balanced' offsets variant distribution issues
        self.classifier = LogisticRegression(C=5.0, max_iter=1000, class_weight='balanced', random_state=42)

    def train_model(self, feature_matrix, labels: list):
        """Fits multi-class weights onto the incoming TF-IDF vector space."""
        self.classifier.fit(feature_matrix, labels)

    def predict_intent(self, query_vector):
        """Evaluates prediction probabilities and returns target classification label."""
        prediction = self.classifier.predict(query_vector)[0]
        probabilities = self.classifier.predict_proba(query_vector)[0]
        max_confidence = max(probabilities)
        
        return prediction, max_confidence