# 📖 Deep Learning & Large Language Models (LLMs)

## 🌐 Introduction
Deep Learning is a branch of Machine Learning that uses **multi-layer neural networks** to learn complex patterns. In NLP, Deep Learning powers **Large Language Models (LLMs)**, which are trained on massive text corpora to understand and generate human-like language.

---

## 🔹 Large Language Models (LLMs)
### Definition
LLMs are advanced AI models trained on billions of words. They can:
- Understand context and meaning.
- Generate coherent responses.
- Perform tasks like translation, summarization, and question answering.

### Evolution
1. **RNN (Recurrent Neural Networks)**  
   - Sequential models that process text word by word.  
   - **Problem:** Struggle with long-term dependencies (forget earlier context).  

2. **Attention Mechanism ("Attention is All You Need")**  
   - Introduced in 2017 by Vaswani et al.  
   - Allows models to focus on relevant words anywhere in the sequence.  
   - Solves RNN’s long dependency problem.

---

# 🔁 Recurrent Neural Networks (RNNs)

### Concept
RNNs maintain a hidden state that carries information forward in a sequence.

### Example 1: Next Word Prediction
Sentence:  
`"I grew up in France. I speak fluent ___"`

- RNN processes sequentially.  
- By the time it predicts the blank, it often forgets *France* → predicts `"English"` instead of `"French"`.  

👉 **Problem:** Vanishing gradients → earlier context fades.

---

### Example 2: Sentiment Classification
Sentence:  
`"The movie was not bad at all"`

- Sequential reading: `"The movie was"` → neutral, `"not"` → negative, `"bad"` → negative.  
- Loses nuance that `"not bad"` = positive.  

👉 Shows RNN’s weakness with long dependencies.

---

# ✨ Attention Mechanism

### Concept
Attention lets the model **focus on relevant words directly**, regardless of position.

### Example 1: Machine Translation
Sentence:  
`"The cat sat on the mat"` → French: `"Le chat s'est assis sur le tapis"`

- Attention aligns `"cat"` → `"chat"`, `"mat"` → `"tapis"`.  
- Directly connects words even if far apart.

---

### Example 2: Summarization
Input:  
`"Apple is buying a UK startup for $1 billion."`  
Output:  
`"Apple plans $1B UK acquisition."`

- Attention highlights `"Apple"`, `"UK startup"`, `"1 billion"`.  
- Ignores filler words.  


---

# ⚡ Transformer Architecture (Attention is All You Need)

Transformers scale attention into a full architecture.

### Encoder-Decoder Structure
1. **Input Embedding (Encoder)**  
   Words → dense vectors.  
   Example: `"dog"` → `[0.12, -0.45, 0.67, ...]`

2. **Positional Encoding**  
   Adds sequence order info.  
   `"dog"` at position 1 ≠ `"dog"` at position 5.

3. **Multi-Head Attention Layer**  
   - **Query (Q):** Current focus word.  
   - **Key (K):** All words in sequence.  
   - **Value (V):** Word meaning.  
   - Attention = weighted sum of values based on Q-K similarity.  

   Example:  
   `"The bank will not lend money to the river project."`  
   - Attention disambiguates `"bank"` → financial institution, not riverbank.

4. **Feedforward Layer (Position-wise Feed-Forward Networks)**  
   After attention, each position is processed independently through a two-layer feedforward network.  
   
   **Structure:**  
   - First layer: Expands dimensionality (e.g., 512 → 2048).  
   - Activation function: ReLU (introduces non-linearity).  
   - Second layer: Projects back to original dimension (2048 → 512).  
   
   **Mathematical formula:**  
   `FFN(x) = ReLU(x * W1 + b1) * W2 + b2`  
   
   **Purpose:**  
   - Adds **non-linearity** to capture complex patterns.  
   - Allows model to learn **task-specific transformations**.  
   - Increases model capacity without increasing attention cost.  
   
   **Example:**  
   Input to FFN: `[0.5, -0.3, 0.8, ...]` (embedding vector)  
   - Expands → `[0.5, 0.2, 1.1, 0.9, ..., 0.0]` (2048 dimensions)  
   - ReLU applied → Negative values become 0  
   - Contracts → `[0.6, -0.2, 0.9, ...]` (512 dimensions, back to original size)  
   
   **Real-world analogy:**  
   If attention is like "reading relevant parts of a document," feedforward is like "thinking deeply about what those parts mean and how they relate to our task."

5. **Masked Multi-Head Attention (Decoder)**  
   Prevents peeking at future words during training.  
   Example: `"I love ___"` → can’t look ahead to `"NLP"`.

6. **Output Embedding (Decoder)**  
   Converts processed vectors back into words.  
   Example: `[0.23, -0.11, 0.89]` → `"French"`

---

### End-to-End Example: Translation
Input: `"Bonjour le monde"`  
- Embedding → `[vectors]`  
- Positional encoding → adds order  
- Multi-head attention → `"Bonjour"` → `"Hello"`, `"monde"` → `"world"`  
- Decoder → outputs `"Hello world"`

---

# 🤗 Hugging Face Ecosystem

## 🌐 What is Hugging Face?
Hugging Face is an open-source platform and library that provides:
- **Pre-trained transformer models** (BERT, GPT, RoBERTa, XLNet, etc.).
- **Tokenizers** for converting text into model-ready input.
- **Pipelines** for common NLP tasks (sentiment analysis, question answering, summarization).
- **Integration** with PyTorch and TensorFlow.
- **Model Hub** for sharing and downloading models.

👉 Purpose: Hugging Face makes **state-of-the-art NLP accessible** without needing to build models from scratch.

---

## 🔹 Difference from Traditional NLP
- **Traditional NLP:** Rule-based, statistical methods (stopword removal, TF-IDF, regex).  
- **Hugging Face:** Transformer-based deep learning models that learn context and semantics from massive datasets.  

---

## 🔹 Fine-Tuning (Deep Explanation)

### What is Fine-Tuning?
Fine-tuning is the process of **adapting a pre-trained model** (like BERT or GPT) to a specific dataset or task.  
- Pre-training: Learns general language knowledge.  
- Fine-tuning: Specializes in your domain (e.g., product reviews, medical text).  

👉 Example: A general BERT model fine-tuned on electronics reviews learns that “battery drain” is negative sentiment.

---

### Why Fine-Tune?
- **Domain Adaptation:** Handles specialized vocabulary.  
- **Task Specialization:** Learns your labels (positive/negative, spam/ham).  
- **Performance Boost:** Outperforms generic models on your dataset.

---

### Workflow
1. **Prepare Data** → Collect labeled examples.  
2. **Load Pre-trained Model** → e.g., `bert-base-uncased`.  
3. **Tokenize Data** → Convert text into input IDs.  
4. **Train (Fine-Tune)** → Adjust weights on your dataset.  
5. **Save Model** → Store fine-tuned version.  
6. **Load Model** → Reload for inference.  
7. **Inference** → Predict on new text.

---

### Example Code
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Step 1: Prepare dataset
df = pd.DataFrame({"text": ["Great laptop!", "Terrible battery"], "label": [1, 0]})
dataset = Dataset.from_pandas(df)

# Step 2: Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Step 3: Tokenize
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)
dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Step 4: Fine-tune
training_args = TrainingArguments(output_dir="./results", num_train_epochs=2, per_device_train_batch_size=8)
trainer = Trainer(model=model, args=training_args, train_dataset=dataset, eval_dataset=dataset)
trainer.train()
```

---

# 🚀 Setup & Run: HuggingFace Products Search

This section provides step-by-step instructions to setup and run the HuggingFace-powered product search system for electronics catalog.

## 📋 Prerequisites

- Python 3.8+
- pip or conda package manager

## 🔧 Installation

### Step 1: Install Required Dependencies

Run the following command in your terminal:

```bash
pip install pandas datasets sentence-transformers faiss-cpu joblib
```

**Package Details:**
- `pandas`: Data manipulation and analysis
- `datasets`: Hugging Face Datasets library
- `sentence-transformers`: Pre-trained models for semantic search
- `faiss-cpu`: Facebook AI Similarity Search (CPU version for semantic indexing)
- `joblib`: Serialization of Python objects

### Step 2: Verify Installation

```bash
python -c "import pandas, datasets, sentence_transformers, faiss, joblib; print('All packages installed successfully!')"
```

---

## 📂 Project Structure

```
llm_and_huggingface/
├── hf_product_indexer.py      # Creates FAISS index from product data
├── hf_product_search.py       # Searches products using semantic similarity
├── electronics_tn.json        # Input data: Electronics products
├── hf_model.pkl              # Saved embedding model (generated)
├── faiss_index.pkl           # Saved FAISS index (generated)
├── hf_products.parquet       # Saved product data (generated)
└── readme.md                 # This file
```

---

## 🏃 Running the Application

### Option 1: Create FAISS Index (Initial Setup)

If you're running this for the first time or want to rebuild the index:

```bash
python hf_product_indexer.py
```

**What happens:**
1. Reads `electronics_tn.json` (product catalog)
2. Extracts and flattens seller location data
3. Combines product fields into unified text
4. Encodes all products using `sentence-transformers/all-MiniLM-L6-v2` model
5. Builds FAISS index for fast semantic search
6. Saves:
   - `hf_model.pkl` - Embedding model
   - `faiss_index.pkl` - FAISS search index
   - `hf_products.parquet` - Processed products

**First run duration:** ~2-5 minutes (depends on dataset size and hardware)

### Option 2: Search Products Using Existing Index

Once the index is built, search for products:

```python
from hf_product_search import HFProductSearch

# Initialize search
search = HFProductSearch()

# Example searches
results1 = search.search("laptop under 50000", top_n=5)
print(results1)

results2 = search.search("samsung mobile between 20000 and 40000", top_n=10)
print(results2)

results3 = search.search("high rating electronics in Chennai", top_n=5)
print(results3)
```

---

## 🔍 Usage Examples

### Example 1: Price Range Search
```python
from hf_product_search import HFProductSearch

search = HFProductSearch()

# Find products under 30,000
results = search.search("laptop under 30000")
print(results)
```

**Output:**
```
       product_name  brand       category  price  city    score
0  Dell Inspiron 15  Dell      Laptops    25000  TN    0.87
1  HP Pavilion 14    HP        Laptops    28000  TN    0.85
...
```

### Example 2: Brand-Specific Search
```python
# Search for specific brands
results = search.search("iphone latest model")
print(results[['product_name', 'brand', 'price', 'city']])
```

### Example 3: Between Price Range
```python
# Find products in a specific price bracket
results = search.search("smartwatch between 5000 and 15000")
print(results)
```

### Example 4: Above Price Threshold
```python
# Find premium products
results = search.search("laptop above 80000")
print(results)
```

---

## 🤖 How It Works

### Indexing Process
1. **Data Loading:** Reads electronics products from JSON
2. **Data Processing:** Flattens nested fields (seller_location)
3. **Text Combination:** Merges product fields into single text
4. **Embedding:** Uses `sentence-transformers/all-MiniLM-L6-v2` to convert text → vectors
5. **Index Creation:** Builds FAISS L2 distance index
6. **Persistence:** Saves model, index, and data as pickle/parquet files

### Search Process
1. **Query Processing:** Normalizes numbers (e.g., "50k" → 50000)
2. **Embedding:** Encodes search query to vector
3. **Semantic Search:** Finds top-k similar products using FAISS
4. **Price Filtering:** Applies price constraints (under/above/between)
5. **Ranking:** Sorts by semantic similarity score
6. **Return Results:** Top N matching products

### Price Intent Detection
- **"under X"** or **"below X":** Max price = X
- **"above X"** or **"greater X":** Min price = X
- **"between X and Y":** Min = X, Max = Y

---

## 📊 Key Features

| Feature | Details |
|---------|---------|
| **Model** | sentence-transformers/all-MiniLM-L6-v2 (fast, lightweight) |
| **Index Type** | FAISS IndexFlatL2 (exact nearest neighbor search) |
| **Search Type** | Semantic + Price-based hybrid search |
| **Number Normalization** | Supports k (1000), l (100k), cr (10M) |
| **Output Fields** | product_name, brand, category, price, city, score |
| **Default Top-N** | 10 results |

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'sentence_transformers'"
**Solution:** Install missing packages
```bash
pip install sentence-transformers faiss-cpu
```

### Issue: "FileNotFoundError: hf_products.parquet"
**Solution:** Run the indexer first
```bash
python hf_product_indexer.py
```

### Issue: "model.pkl not found" or "index.pkl not found"
**Solution:** Rebuild the index
```bash
rm *.pkl *.parquet  # Clean old files
python hf_product_indexer.py
```

### Issue: Slow search performance
**Solution:** 
- Use fewer results: `top_n=5` instead of `top_n=50`
- Use GPU version of FAISS: `pip install faiss-gpu`

---

## 📝 Notes

- **First Run:** Downloads embedding model (~40MB) on first use
- **Index Rebuilding:** Only needed if product data changes
- **Memory Usage:** ~500MB-1GB depending on dataset size
- **CPU Search:** Uses CPU-optimized FAISS (no GPU needed)

---

## ✅ Testing

Quick test to verify everything works:

```bash
# Test 1: Build index
python hf_product_indexer.py

# Test 2: Search
python -c "
from hf_product_search import HFProductSearch
search = HFProductSearch()
results = search.search('laptop under 50000', top_n=3)
print('Test successful! Found', len(results), 'products')
print(results)
"
```

---
