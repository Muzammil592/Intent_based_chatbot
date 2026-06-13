import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from spellchecker import SpellChecker

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

class AdvancedPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.spell = SpellChecker()
        self.stopwords = {'is', 'the', 'a', 'an', 'are', 'am', 'to', 'at', 'in', 'on', 'for', 'of', 'do', 'does'}
        
# Optimized Day 2 Tuning Matrix
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), 
            max_features=1000,
            sublinear_tf=True,  # Scaling down logarithmic term frequencies to prevent skewing
            min_df=1            # Ensure single words are captured completely
        )
        # Standard contraction mapping matrix
        self.contractions = {
            "i'm": "i am", "can't": "cannot", "don't": "do not", "what's": "what is",
            "you're": "you are", "it's": "it is", "she's": "she is", "he's": "he is"
        }

    def expand_contractions(self, text: str) -> str:
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        return text

    def clean_text(self, text: str, fix_spelling: bool = False) -> str:
        """Advanced cleaning pipe: Contractions -> Regex -> Spelling Correction -> Lemmatization"""
        text = text.lower().strip()
        text = self.expand_contractions(text)
        text = re.sub(r'[^\w\s]', '', text) # Strip punctuation
        
        tokens = nltk.word_tokenize(text)
        processed_tokens = []
        
        for token in tokens:
            # Execute spelling check only on live user queries to keep training data un-warped
            if fix_spelling and token not in self.stopwords:
                corrected = self.spell.correction(token)
                token = corrected if corrected else token
                
            if token not in self.stopwords:
                lemma = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemma)
                
        return " ".join(processed_tokens)

    def fit_transform_matrix(self, text_list: list):
        cleaned_corpus = [self.clean_text(text, fix_spelling=False) for text in text_list]
        return self.vectorizer.fit_transform(cleaned_corpus)

    def transform_query(self, query: str):
        cleaned_query = self.clean_text(query, fix_spelling=True)
        return self.vectorizer.transform([cleaned_query])