import pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformer
import faiss
import joblib

class HFProductIndexer:
    def __init__(self, file_path):
        print("Indexing started...")
        self.file_path = file_path
        self.data = pd.read_json(self.file_path, lines=True)

        # Flatten seller_location
        self.data['city'] = self.data['seller_location'].apply(lambda x: x['city'])
        self.data['lat'] = self.data['seller_location'].apply(lambda x: x['lat'])
        self.data['lon'] = self.data['seller_location'].apply(lambda x: x['lon'])

        # Combine product fields into one text field
        self.data['product_text'] = self.data.apply(lambda row: f"""
            Product: {row['product_name']}
            Category: {row['category']}
            Brand: {row['brand']}
            Review: {row['review_text']}
            Price: {row['price']}
            City: {row['city']}
            Seller Rating: {row['seller_rating']}""", axis=1)

        # Hugging Face embedding model
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.embeddings = self.model.encode(self.data['product_text'].tolist(), convert_to_tensor=False)

        # Build FAISS index
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

        # Save artifacts
        joblib.dump(self.model, "hf_model.pkl")
        joblib.dump(self.index, "faiss_index.pkl")
        self.data.to_parquet("hf_products.parquet", index=False)
        print("Indexing completed successfully.")

hf_indexer = HFProductIndexer('electronics_tn.json')
# print(hf_indexer.data.head())