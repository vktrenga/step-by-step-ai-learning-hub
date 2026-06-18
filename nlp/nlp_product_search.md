
# 📖 NLP-Based Product Search System Documentation

## 🌐 Overview
This system is designed to **index product data** using Natural Language Processing (NLP) techniques and then allow **semantic search** with natural language queries. It consists of two main components:

1. **ProductIndexer** → Prepares and enriches product data with NLP features.  
2. **ProductSearch** → Processes user queries, applies filters, and retrieves the most relevant products.

---

## 🏗️ Component 1: ProductIndexer

### Purpose
Transforms raw product data into a structured, searchable format enriched with NLP features.

### Steps
1. **Data Loading**
   - Reads JSON product data.
   - Extracts seller location details (city, lat, lon).

2. **Data Combination**
   - Creates a unified text field (`product_text`) combining product attributes:
     - Name, category, brand, review, price, city, seller rating.

3. **Text Preprocessing**
   - Lowercasing, punctuation removal.
   - Stopword removal (except “not” for sentiment accuracy).
   - Lemmatization for dictionary form words.

4. **Vectorization**
   - Uses **TF-IDF with n-grams (1–2)**.
   - Builds a **cosine similarity matrix** for product similarity.

5. **POS & NER Analysis**
   - Uses **spaCy** for:
     - POS tagging (grammatical roles).
     - Named Entity Recognition (brands, locations, dates).

6. **Sentiment Analysis**
   - Uses **VADER** to compute sentiment scores from reviews.

7. **Topic Modeling**
   - Uses **LDA (Latent Dirichlet Allocation)** to discover hidden themes in reviews.

8. **Classifier Training**
   - Labels reviews as positive/negative based on rating.
   - Trains a **Logistic Regression classifier**.
   - Saves model (`review_classifier.pkl`).

9. **Data Storage**
   - Saves vectorizer, TF-IDF matrix, and enriched product data (`products.parquet`).

---

## 🔍 Component 2: ProductSearch

### Purpose
Processes natural language queries to retrieve relevant products with filters for **price, location, and similarity**.

### Steps
1. **Initialization**
   - Loads stored product data (`products.parquet`).
   - Loads vectorizer and TF-IDF matrix.

2. **Preprocessing Queries**
   - Lowercasing, punctuation removal.
   - Stopword removal (except “not”).
   - Lemmatization.

3. **Number Normalization**
   - Converts shorthand:
     - `50k` → `50000`
     - `1L` → `100000`
     - `2 Cr` → `20000000`

4. **Price Intent Detection**
   - Uses **RapidFuzz** to detect intent:
     - `"below 1L"` → max price filter.
     - `"above 50k"` → min price filter.
     - `"between 30k and 50k"` → range filter.

5. **City Extraction**
   - Matches city names from dataset against query.

6. **Query Cleaning**
   - Removes city names and numbers.
   - Leaves descriptive product terms.

7. **Vectorization & Similarity**
   - Converts query into TF-IDF vector.
   - Computes cosine similarity with product vectors.
   - Assigns similarity score.

8. **Filtering**
   - Applies city filter.
   - Applies price filter (min/max/between).

9. **Sorting & Returning Results**
   - Sorts products by similarity score.
   - Returns top `n` results with:
     - Product name, brand, category, price, city, similarity score.

---

## 🖼️ Example Workflow

**Query:**  
`"Best budget Lenovo laptop below 1L and above 50k in Coimbatore"`

**Process:**
1. Normalize → `"below 100000 and above 50000 in Coimbatore"`.
2. Detect intent → min=50000, max=100000.
3. Extract city → `"Coimbatore"`.
4. Clean query → `"best budget lenovo laptop"`.
5. Vectorize & compare → similarity scores.
6. Filter → only Coimbatore laptops in ₹50k–₹1L range.
7. Return → top ranked Lenovo laptops.

---

## 🚀 Key Features
- **Semantic Search** → Finds relevant products beyond exact keyword matches.  
- **Price Awareness** → Understands shorthand (k, L, Cr) and intent (above, below, between).  
- **Location Awareness** → Filters by city names.  
- **Sentiment & Topics** → Enriches product insights with review analysis.  
- **Reusable Pipeline** → Indexed data stored for fast search.  

---

## 📊 System Architecture (Simplified Flow)

```
Raw Product Data (JSON)
        ↓
ProductIndexer
  - Preprocessing
  - TF-IDF Vectorization
  - POS/NER, Sentiment, Topics
  - Classifier Training
  - Store Data
        ↓
Stored Data (Parquet + Models)
        ↓
ProductSearch
  - Query Preprocessing
  - Number Normalization
  - Price & City Filters
  - TF-IDF + Cosine Similarity
  - Sorted Results
        ↓
Top Product Matches
```

