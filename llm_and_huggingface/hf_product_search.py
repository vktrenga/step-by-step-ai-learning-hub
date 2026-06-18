import re
import pandas as pd
import joblib
from sentence_transformers import util

class HFProductSearch:
    def __init__(self):
        # Load stored files
        self.data = pd.read_parquet("hf_products.parquet")
        self.model = joblib.load("hf_model.pkl")
        self.index = joblib.load("faiss_index.pkl")

    def normalize_numbers(self, text):
        text = re.sub(r'(\d+)(k|K)', lambda m: str(int(m.group(1)) * 1000), text)
        text = re.sub(r'(\d+(\.\d+)?)(l|L)', lambda m: str(float(m.group(1)) * 100000), text)
        text = re.sub(r'(\d+(\.\d+)?)(cr|Cr|CR)', lambda m: str(float(m.group(1)) * 10000000), text)
        return text

    def search(self, query, top_n=10):
        query = self.normalize_numbers(query.lower())

        # Encode query
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # Semantic similarity
        scores = util.cos_sim(query_embedding, self.model.encode(self.data['product_text'].tolist(), convert_to_tensor=True))[0].cpu().numpy()
        self.data['score'] = scores

        # Price intent detection
        numbers = [int(n) for n in re.findall(r'\d+', query)]
        min_price, max_price = None, None
        if "under" in query or "below" in query:
            max_price = numbers[0] if numbers else None
        elif "above" in query or "greater" in query:
            min_price = numbers[0] if numbers else None
        elif "between" in query and len(numbers) >= 2:
            min_price, max_price = numbers[0], numbers[1]

        # Apply filters
        filtered = self.data.copy()
        if min_price is not None:
            filtered = filtered[filtered['price'] >= min_price]
        if max_price is not None:
            filtered = filtered[filtered['price'] <= max_price]

        # Sort by semantic score
        results = filtered.sort_values(by='score', ascending=False)
        return results[['product_name','brand','category','price','city','score']].head(top_n)

# ---------------------------------
# RUN SEARCH
# ---------------------------------
if __name__ == "__main__":
    print("Running HF Product Search...")
    search_engine = HFProductSearch()
    results = search_engine.search(
        "Best budget Lenovo laptop below 1L and above 50k in Coimbatore"
    )
    print(results)