import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required statistical NLTK corpora inside the script runtime
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        # Basic English stopwords list hardcoded to keep processing pure
        self.stopwords = {'is', 'the', 'a', 'an', 'are', 'am', 'to', 'at', 'in', 'on', 'for', 'of', 'do', 'does'}
        self.vectorizer = TfidfVectorizer()

    def clean_text(self, text: str) -> str:
        """Removes punctuation, lowers case parameters, tokenizes and lemmatizes words."""
        # Clean special characters out using Regex
        text = re.sub(r'[^\w\s]', '', text.lower().strip())
        
        # Tokenization via NLTK string splittings
        tokens = nltk.word_tokenize(text)
        
        # Filter stopwords and execute WordNet Lemmatization
        processed_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stopwords
        ]
        
        return " ".join(processed_tokens)

    def fit_transform_matrix(self, text_list: list):
        """Fits vocabulary space and outputs sparse TF-IDF feature weight matrix."""
        cleaned_corpus = [self.clean_text(text) for text in text_list]
        return self.vectorizer.fit_transform(cleaned_corpus)

    def transform_query(self, query: str):
        """Transforms a live user string based on pre-fitted vocabulary structures."""
        cleaned_query = self.clean_text(query)
        return self.vectorizer.transform([cleaned_query])