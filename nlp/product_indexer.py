import pandas as pd
import joblib
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression

# Extra NLP libraries
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gensim import corpora, models

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")
analyzer = SentimentIntensityAnalyzer()

class ProductIndexer:
    def __init__(self, file_path):
        print("Started")
        self.file_path = file_path
        self.data = self.read_data()
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.vectorizer = None
        self.classifier = None

        # Pipeline steps
        self.convert_data()
        self.combine_data()
        self.process_data()
        self.vectorize_data()
        self.pos_ner_analysis()
        self.sentiment_analysis()
        self.topic_modeling()   
        self.train_classifier()
        self.store_data()
        print("Pipeline completed successfully.")

    def read_data(self):
        print("read_data")
        return pd.read_json(self.file_path, lines=True)

    def convert_data(self):
        print("convert_data")
        self.data['city'] = self.data["seller_location"].apply(lambda x: x["city"])
        self.data['lat'] = self.data["seller_location"].apply(lambda x: x["lat"])
        self.data['lon'] = self.data["seller_location"].apply(lambda x: x["lon"])

    def combine_data(self):
        print("combine_data")
        self.data["product_text"] = self.data.apply(lambda row: f"""
            Product: {row['product_name']}
            Category: {row['category']}
            Brand: {row['brand']}
            Review: {row['review_text']}
            Price: {row['price']}
            City: {row['city']}
            Seller Rating: {row['seller_rating']}""", axis=1)

    def process_data(self):
        print("process_data")
        # Lowercase
        self.data['product_text'] = self.data['product_text'].str.lower()

        # Remove punctuation
        self.data['product_text'] = self.data['product_text'].str.replace(r'[^\w\s]', '', regex=True)

        # Remove stop words (keep "not")
        stop_words = set(stopwords.words('english'))
        if "not" in stop_words:
            stop_words.remove("not")
        self.data['product_text'] = self.data['product_text'].apply(
            lambda text: " ".join([word for word in word_tokenize(text) if word not in stop_words])
        )

        # Lemmatization
        self.data['product_text'] = self.data['product_text'].apply(
            lambda text: " ".join([lemmatizer.lemmatize(token) for token in word_tokenize(text)])
        )

    def vectorize_data(self):
        print("vectorize_data")
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.data['product_text'])
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

    def pos_ner_analysis(self):
        print("pos_ner_analysis")
        self.data['pos_tags'] = self.data['review_text'].apply(
            lambda text: [(token.text, token.pos_) for token in nlp(text)]
        )
        self.data['entities'] = self.data['review_text'].apply(
            lambda text: [(ent.text, ent.label_) for ent in nlp(text).ents]
        )

    def sentiment_analysis(self):
        print("sentiment_analysis")
        self.data['sentiment'] = self.data['review_text'].apply(
            lambda text: analyzer.polarity_scores(text)['compound']
        )

    def topic_modeling(self, num_topics=2):
        print("topic_modeling")
        tokenized_reviews = [word_tokenize(text.lower()) for text in self.data['review_text']]
        dictionary = corpora.Dictionary(tokenized_reviews)
        corpus = [dictionary.doc2bow(text) for text in tokenized_reviews]
        lda = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
        self.data['topics'] = [lda[dictionary.doc2bow(text)] for text in tokenized_reviews]

    def train_classifier(self):
        print("train_classifier")
        labels = (self.data['review_rating'] >= 3).astype(int)  # 1=Positive, 0=Negative
        clf = LogisticRegression()
        clf.fit(self.tfidf_matrix, labels)
        self.classifier = clf
        joblib.dump(clf, 'review_classifier.pkl')

    def store_data(self):
        print("store_data")
        joblib.dump(self.vectorizer, 'vectorizer.pkl')
        joblib.dump(self.tfidf_matrix, 'tfidf_matrix.pkl')
        self.data.to_parquet('products.parquet', index=False)

# Run pipeline
new_indexer = ProductIndexer('electronics_tn.json')
print(new_indexer.data.head())
