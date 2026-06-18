import re
import joblib
import pandas as pd

from rapidfuzz import fuzz
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
if "not" in stop_words:
    stop_words.remove("not")


class ProductSearch:

    def __init__(self):
        # -----------------------------
        # LOAD STORED FILES
        # -----------------------------
        self.data = pd.read_parquet('products.parquet')
        self.vectorizer = joblib.load('vectorizer.pkl')
        self.tfidf_matrix = joblib.load('tfidf_matrix.pkl')

    # ---------------------------------
    # NLP PREPROCESSING
    # ---------------------------------
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in stop_words]
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return " ".join(tokens)


    def normalize_numbers(self, text):
        # Normalize K/k → thousands
        text = re.sub(r'(\d+)(k|K)', lambda m: str(int(m.group(1)) * 1000), text)

        # Normalize L/l → lakhs
        text = re.sub(r'(\d+(\.\d+)?)(l|L)', 
                    lambda m: str(float(m.group(1)) * 100000), text)

        # Normalize Cr/cr → crores
        text = re.sub(r'(\d+(\.\d+)?)(cr|Cr|CR)', 
                    lambda m: str(float(m.group(1)) * 10000000), text)

        return text

    # ---------------------------------
    # SEARCH
    # ---------------------------------
    def search(self, query, top_n=10):
        query_lower = self.normalize_numbers(query.lower())
        min_price, max_price = None, None

        # -----------------------------
        # PRICE INTENT DETECTION
        # -----------------------------
        PRICE_PATTERNS = {
            "max": ["below", "under", "less than", "upto"],
            "min": ["above", "greater than", "more than", "minimum"],
            "between": ["between"]
        }

        detected_intent, best_score = None, 0
        for intent, keywords in PRICE_PATTERNS.items():
            for keyword in keywords:
                score = fuzz.partial_ratio(keyword, query_lower)
                if score > best_score:
                    best_score = score
                    detected_intent = intent

        # -----------------------------
        # EXTRACT NUMBERS
        # -----------------------------
        numbers = [int(n) for n in re.findall(r'\d+', query_lower)]

        if detected_intent == "between" and len(numbers) >= 2:
            min_price, max_price = numbers[0], numbers[1]
        elif detected_intent == "max" and len(numbers) >= 1:
            max_price = numbers[0]
        elif detected_intent == "min" and len(numbers) >= 1:
            min_price = numbers[0]

        # -----------------------------
        # CITY EXTRACTION (MULTIPLE)
        # -----------------------------
        matched_cities = [
            c for c in self.data['city'].unique()
            if c.lower() in query_lower
        ]

        # -----------------------------
        # CLEAN QUERY
        # -----------------------------
        clean_query = query_lower
        for c in matched_cities:
            clean_query = clean_query.replace(c.lower(), '')
        clean_query = re.sub(r'\d+', '', clean_query)

        # -----------------------------
        # NLP PROCESSING
        # -----------------------------
        processed_query = self.preprocess_text(clean_query)

        # -----------------------------
        # QUERY VECTOR
        # -----------------------------
        query_vector = self.vectorizer.transform([processed_query])

        # -----------------------------
        # COSINE SIMILARITY
        # -----------------------------
        scores = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        self.data['score'] = scores
        filtered_data = self.data.copy()

        # -----------------------------
        # APPLY CITY FILTER
        # -----------------------------
        if matched_cities:
            filtered_data = filtered_data[
                filtered_data['city'].str.lower().isin([c.lower() for c in matched_cities])
            ]

        # -----------------------------
        # APPLY PRICE FILTERS
        # -----------------------------
        if min_price is not None:
            filtered_data = filtered_data[filtered_data['price'] >= min_price]
        if max_price is not None:
            filtered_data = filtered_data[filtered_data['price'] <= max_price]

        # -----------------------------
        # SORT RESULTS
        # -----------------------------
        results = filtered_data.sort_values(by='score', ascending=False)

        # -----------------------------
        # RETURN TOP RESULTS
        # -----------------------------
        return results[['product_name', 'brand', 'category', 'price', 'city', 'score']].head(top_n)


# ---------------------------------
# RUN SEARCH
# ---------------------------------
if __name__ == "__main__":
    search_engine = ProductSearch()
    results = search_engine.search(
        "Best budget Lenovo laptop below 1L and above 50k in Coimbatore"
    )
    print(results)
