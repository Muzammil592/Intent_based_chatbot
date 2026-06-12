from sklearn.linear_model import LogisticRegression

class IntentClassifier:
    def __init__(self):
        # High C parameter handles classification penalties strictly
        self.classifier = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

    def train_model(self, feature_matrix, labels: list):
        """Fits multi-class weights onto the incoming TF-IDF vector space."""
        self.classifier.fit(feature_matrix, labels)

    def predict_intent(self, query_vector):
        """Evaluates prediction probabilities and returns target classification label."""
        prediction = self.classifier.predict(query_vector)[0]
        # Calculate maximum confidence score output
        probabilities = self.classifier.predict_proba(query_vector)[0]
        max_confidence = max(probabilities)
        
        return prediction, max_confidence