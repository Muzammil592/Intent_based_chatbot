import re
import yaml
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from spellchecker import SpellChecker

class AdvancedPreprocessor:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['model']
            
        self.lemmatizer = WordNetLemmatizer()
        self.spell = SpellChecker()
        self.stopwords = {'is', 'the', 'a', 'an', 'are', 'am', 'to', 'at', 'in', 'on', 'for', 'of', 'do', 'does'}
        
        self.vectorizer = TfidfVectorizer(
            ngram_range=tuple(self.config['ngram_range']), 
            max_features=self.config['max_features'],
            sublinear_tf=True,
            min_df=1
        )
        self.contractions = {
            "i'm": "i am", "can't": "cannot", "don't": "do not", "what's": "what is",
            "you're": "you are", "it's": "it is"
        }

    def expand_contractions(self, text: str) -> str:
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        return text

    def clean_text(self, text: str, fix_spelling: bool = False) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower().strip()
        text = self.expand_contractions(text)
        text = re.sub(r'[^\w\s]', '', text)
        
        tokens = nltk.word_tokenize(text)
        processed_tokens = []
        for token in tokens:
            if fix_spelling and token not in self.stopwords:
                corrected = self.spell.correction(token)
                token = corrected if corrected else token
            if token not in self.stopwords:
                processed_tokens.append(self.lemmatizer.lemmatize(token))
        return " ".join(processed_tokens)

    def fit_transform_matrix(self, text_list: list):
        cleaned_corpus = [self.clean_text(text, fix_spelling=False) for text in text_list]
        return self.vectorizer.fit_transform(cleaned_corpus)

    def transform_query(self, query: str):
        cleaned_query = self.clean_text(query, fix_spelling=True)
        return self.vectorizer.transform([cleaned_query])