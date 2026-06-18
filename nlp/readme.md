# 📖 Natural Language Processing (NLP)

## 🌐 What is NLP?
Natural Language Processing (NLP) is a field of Artificial Intelligence (AI) that enables computers to understand, interpret, and generate human language. It bridges the gap between human communication (words, sentences, meaning) and machine understanding (data, algorithms, models).

- **Purpose:**  
  - Make machines understand human language.  
  - Enable tasks like translation, sentiment analysis, chatbots, search engines, and document classification.  
  - Extract insights from large volumes of text (e.g., social media, customer feedback, research papers).

- **Where is NLP Used?**  
  - **Search Engines:** Ranking results based on query meaning.  
  - **Chatbots & Virtual Assistants:** Understanding user intent.  
  - **Healthcare:** Extracting medical information from patient records.  
  - **Finance:** Detecting fraud or analyzing reports.  
  - **Social Media:** Sentiment analysis of posts.  
  - **Business:** Spam detection, document classification, recommendation systems.

---

## 📝 Text Processing
Text preprocessing is the **foundation of NLP**. It cleans and structures raw text into a usable format.

- **Lowercase**  
  Ensures uniformity by converting all text to lowercase.  
  *Example:* `"Hello World"` → `"hello world"`

- **Removing Stop Words**  
  Eliminates common words that don't add meaning.  
  *Example:* `"This is a good book"` → `"good book"`

- **Regular Expression (Regex)**  
  Cleans text using patterns.  
  *Example:* Remove digits: `"Book123"` → `"Book"`

- **Tokenization**  
  Splits text into smaller units.  
  - Word Tokenization: `"I love NLP"` → `["I", "love", "NLP"]`  
  - Sentence Tokenization: `"I love NLP. It is powerful."` → `["I love NLP.", "It is powerful."]`

- **Stemming**  
  Reduces words to their root form (may not be valid words).  
  *Example:* `"running", "runs"` → `"run"`

- **Lemmatization**  
  Converts words to dictionary form using context.  
  *Example:* `"better"` → `"good"`

- **N-Grams**  
  Sequence of *n* words.  
  *Example:* For bigrams (n=2): `"I love NLP"` → `["I love", "love NLP"]`

---

## 🔎 Identifying Parts of Speech & Named Entities
Understanding grammar and meaning is crucial for deeper analysis.

- **POS Tagging**  
  Labels words with grammatical roles.  
  *Example:* `"The cat sleeps"` → `[("The", DET), ("cat", NOUN), ("sleeps", VERB)]`

- **Named Entity Recognition (NER)**  
  Identifies entities like names, places, dates.  
  *Example:* `"Apple was founded in California in 1976"` → `[("Apple", ORG), ("California", LOC), ("1976", DATE)]`

---

## 😊 Sentiment Analysis
Determines emotional tone in text.

- **Rule-Based (TextBlob, Vader)**  
  Uses predefined lexicons.  
  *Example (Vader):* `"I love this movie!"` → Positive sentiment score.

- **Pre-trained Transformer Models**  
  Deep learning models like BERT or RoBERTa.  
  *Example:* `"The product is terrible"` → Negative sentiment (via fine-tuned BERT).

---

## 📊 Vectorizing Text
Converts text into numerical features for machine learning.

- **Bag of Words (BoW)**  
  Represents text as word counts.  
  *Example:* `"I love NLP"` → `{I:1, love:1, NLP:1}`

- **TF-IDF (Term Frequency – Inverse Document Frequency)**  
  Weighs words by importance across documents.  
  *Example:* `"NLP"` appears rarely → higher weight than `"the"`.

---

## 📚 Topic Modeling
Discovers hidden themes in large text collections.

- **What is Topic Modeling?**  
  Unsupervised method to group documents by topics.

- **When to Use?**  
  Large text datasets where manual classification is impractical.

- **LDA (Latent Dirichlet Allocation)**  
  Probabilistic model that assigns topics to words.  
  *Example:* News articles → Topic 1: Politics, Topic 2: Sports.

- **LSA (Latent Semantic Analysis)**  
  Uses matrix decomposition (SVD) to find word associations.  
  *Example:* `"doctor, hospital, patient"` cluster together → Health topic.

---

## 🏷️ Custom Text Classifiers
Supervised models trained for specific tasks.

- **Logistic Regression**  
  Linear model for classification.  
  *Example:* Spam detection: `"Free money!!!"` → Spam.

- **Naive Bayes**  
  Probabilistic classifier assuming independence.  
  *Example:* `"Discount offer"` → Spam (based on word probabilities).

- **Linear SVM (Support Vector Machine)**  
  Finds hyperplane to separate classes.  
  *Example:* Classify reviews: Positive vs Negative.

---

## 🔄 NLP Process Flow (End-to-End)
1. **Text Collection** → Gather raw text (documents, tweets, reviews).  
2. **Text Processing** → Clean and normalize text.  
3. **Feature Extraction** → Convert text into vectors (BoW, TF-IDF, embeddings).  
4. **Analysis** → Apply POS tagging, NER, sentiment analysis, topic modeling.  
5. **Modeling** → Train classifiers (Logistic Regression, Naive Bayes, SVM, Transformers).  
6. **Evaluation** → Measure accuracy, precision, recall, F1-score.  
7. **Deployment** → Integrate into applications (chatbots, search engines, recommendation systems).

---

# 🧠 Core LLM Concepts

## 1. Tokenization

### What is Tokenization?
Tokenization is the process of breaking down text into smaller units called **tokens** that models can understand.  
- A token ≈ a word, subword, or character.  
- Models work with **token IDs** (integers), not raw text.  
- Different models use different tokenizers.

### Tokenization Example

**Input Text:**
```
"I love NLP!"
```

**Word-level Tokenization:**
```
["I", "love", "NLP", "!"]
→ [1, 2523, 15003, 999]  (token IDs)
```

**Subword Tokenization (BPE - Byte Pair Encoding):**
```
["I", "love", "NLP", "!"]
→ Often splits rare words:
"NLP" → ["NL", "##P"]  (## indicates continuation)
```

### Why Tokenization Matters
- **Vocabulary size:** Determines model size. Typical: 30K-100K tokens.
- **Out-of-vocabulary (OOV) words:** Subword tokenization handles unknown words.
- **Language support:** Different tokenizers for different languages.

### Code Example
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "I love Natural Language Processing!"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)

print("Tokens:", tokens)
# Output: ['i', 'love', 'natural', 'language', 'processing', '!']

print("Token IDs:", token_ids)
# Output: [101, 1045, 2572, 3721, 1570, 3231, 999, 102]
```

---

## 2. Embedding

### What is Embedding?
An embedding is a **dense vector representation** of a token or word.  
- Maps discrete tokens → continuous high-dimensional space.  
- Captures semantic meaning.  
- Example: `"dog"` → `[0.2, -0.5, 0.8, 0.1, ...]` (768 dimensions for BERT)

### Word Embeddings vs. Contextual Embeddings

**Traditional Word Embeddings (Word2Vec, GloVe):**
```
"bank" → [0.3, -0.2, 0.7, ...]  (same vector always)
```
- **Problem:** Ignores context. `"bank"` has different meanings.

**Contextual Embeddings (BERT, GPT):**
```
"I deposit money at the bank" → [0.5, -0.1, 0.9, ...]
"The river bank is beautiful" → [0.2, -0.4, 0.6, ...]  (different!)
```
- **Advantage:** Same word has different embeddings depending on context.

### Embedding Space Visualization

```
Vector Space (2D visualization):
        
        "queen"
          ↓
          ●
         /
        /  (relationship captured)
       /
"king" ● ——————— "man"
       \
        \
         ●
      "woman"

Property:
vector("king") - vector("man") ≈ vector("queen") - vector("woman")
(King to man ≈ Queen to woman)
```

### Code Example
```python
from transformers import AutoTokenizer, AutoModel
import torch

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

text = "I love NLP"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

embeddings = outputs.last_hidden_state
print("Embedding shape:", embeddings.shape)
# Output: torch.Size([1, 6, 768])
# 1 batch, 6 tokens, 768 dimensions

# Get sentence-level embedding (average pooling)
sentence_embedding = embeddings.mean(dim=1)
print("Sentence embedding shape:", sentence_embedding.shape)
# Output: torch.Size([1, 768])
```

### Use Cases
- **Semantic similarity:** Find similar sentences using embeddings.
- **Clustering:** Group documents by meaning.
- **Search:** Find relevant documents via embedding distance.

---

## 3. Context Window

### What is Context Window?
The **maximum number of tokens** a model can process at once.  
- Also called **sequence length** or **max tokens**.  
- Determines how much text the model "sees" simultaneously.  

### Context Window Examples

**GPT-3.5 Turbo:**
- Context: 4,096 tokens
- ≈ 3,000 words (1 token ≈ 0.75 words)

**GPT-4:**
- Short: 8,192 tokens
- Long: 128,000 tokens

**BERT:**
- 512 tokens (fixed)

### Implications

**Text too long:**
```
Input: "Here's a 10,000 word document..." (50,000 tokens)
Model sees: First 4,096 tokens only
Result: Loses tail context
```

**Solutions:**
1. **Chunking:** Split into smaller pieces
2. **Sliding window:** Overlap chunks
3. **Summarization:** Reduce document size
4. **Long-context models:** Use models with larger windows

### Code Example
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

long_text = "The quick brown fox jumps over the lazy dog. " * 1000

tokens = tokenizer.encode(long_text)
print(f"Document length: {len(tokens)} tokens")
print(f"Model context window: {tokenizer.model_max_length} tokens")

if len(tokens) > tokenizer.model_max_length:
    print(f"⚠️  Warning: Document exceeds context window by {len(tokens) - tokenizer.model_max_length} tokens")
    # Truncate to context window
    truncated = tokens[:tokenizer.model_max_length]
```

---

## 4. Model Parameters

### What are Parameters?
**Parameters** are trainable weights in a neural network.  
- Each layer has weights and biases.  
- More parameters → more capacity but slower inference.
- Measured in millions (M), billions (B), or trillions (T).

### Parameter Examples

| Model | Parameters | Training Size |
|-------|-----------|---------------|
| BERT | 110M | 12 GB |
| GPT-2 | 1.5B | 50 GB |
| GPT-3 | 175B | 570 GB |
| LLaMA-2 | 70B | 200+ GB |
| Falcon | 180B | 500+ GB |

### Impact of Parameters

**Fewer Parameters (110M):**
- ✅ Fast inference
- ✅ Small model size
- ❌ Lower accuracy

**More Parameters (175B):**
- ✅ Better performance
- ✅ Few-shot learning
- ❌ Slow inference, expensive

### Code Example: Counting Parameters
```python
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-uncased")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = total_params - trainable_params

print(f"Total parameters: {total_params:,}")
# Output: Total parameters: 109,482,240

print(f"Trainable: {trainable_params:,}")
print(f"Frozen: {frozen_params:,}")

# Estimate memory (assuming float32 = 4 bytes per param)
memory_mb = (total_params * 4) / (1024 ** 2)
print(f"Approx memory: {memory_mb:.2f} MB")
```

---

## 5. Temperature

### What is Temperature?
Controls the **randomness/creativity** of model outputs.  
- Range: 0.0 to 2.0 (typically)
- Lower = more deterministic
- Higher = more random

### Temperature Examples

**Temperature = 0 (Deterministic):**
```
Prompt: "The capital of France is"
Output: "Paris"  (always same)
```
- **Use case:** Factual answers, Q&A, code generation

**Temperature = 0.7 (Balanced):**
```
Prompt: "Tell me a story about a robot"
Output: (varied but coherent)
```
- **Use case:** General chat, content creation

**Temperature = 1.5 (Creative):**
```
Prompt: "Tell me a story about a robot"
Output: (highly varied, sometimes nonsensical)
```
- **Use case:** Creative writing, brainstorming

### Mathematical Explanation
Temperature affects probability distribution:
```
P(token) = exp(logits / temperature) / sum(exp(logits / temperature))

Low temperature (0.1): Makes high-probability tokens even more likely
High temperature (2.0): Flattens distribution, all tokens more equal
```

### Code Example
```python
from transformers import pipeline

# Low temperature = deterministic
generator_factual = pipeline("text-generation", 
    model="gpt2", 
    temperature=0.1, 
    max_length=50
)

# High temperature = creative
generator_creative = pipeline("text-generation", 
    model="gpt2", 
    temperature=1.5, 
    max_length=50
)

prompt = "Once upon a time, there was a"

print("Deterministic (T=0.1):")
print(generator_factual(prompt))

print("\nCreative (T=1.5):")
print(generator_creative(prompt))
```

---

## 6. Top-K and Top-P (Nucleus) Sampling

### Top-K Sampling
Only sample from the **K most likely tokens**.

**Example with K=3:**
```
Probabilities: A(0.7), B(0.15), C(0.1), D(0.03), E(0.02)
Top-3: A, B, C
Renormalize: A(0.778), B(0.167), C(0.111)
Sample from: A, B, C only
```

### Top-P (Nucleus) Sampling
Sample from tokens whose **cumulative probability ≤ P**.

**Example with P=0.9:**
```
Probabilities: A(0.7), B(0.15), C(0.1), D(0.03), E(0.02)
Cumulative: A(0.7), A+B(0.85), A+B+C(0.95) > 0.9
Top-P includes: A, B, C
```

### When to Use
- **Top-K=50:** Good default for balance
- **Top-P=0.9:** Nucleus sampling, prevents absurdities
- **Combine:** Top-K=50, Top-P=0.9 for best results

### Code Example
```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

text = "The future of AI is"

# Top-K sampling
output_topk = generator(
    text,
    max_length=30,
    top_k=50,
    do_sample=True
)

# Top-P sampling
output_topp = generator(
    text,
    max_length=30,
    top_p=0.9,
    do_sample=True
)

# Greedy (no sampling)
output_greedy = generator(
    text,
    max_length=30,
    do_sample=False
)
```

---

## 7. Attention Heads

### What are Attention Heads?
**Attention heads** allow the model to attend to different parts of input simultaneously.  
- Multi-head attention = multiple parallel attention mechanisms.  
- Each head can focus on different relationships.

### Attention Head Example

**Input:** `"The cat sat on the mat"`

**Head 1:** Focuses on subject-verb relationships
```
"cat" → "sat"
```

**Head 2:** Focuses on object relationships
```
"sat" → "mat"
```

**Head 3:** Focuses on prepositional relationships
```
"on" → "the"
```

### Number of Heads by Model

| Model | Attention Heads | Hidden Dim | Head Dim |
|-------|-----------------|-----------|----------|
| BERT | 12 | 768 | 64 |
| GPT-2 | 12 | 768 | 64 |
| GPT-3 | 96 | 12,288 | 128 |

Formula: `Head Dimension = Hidden Dimension / Number of Heads`

### Code Example: Visualizing Attention
```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased", output_attentions=True)

text = "The cat sat on the mat"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

# Access attention weights (layer, batch, heads, seq_len, seq_len)
attention = outputs.attentions
print(f"Attention shape: {attention[0].shape}")
# Output: torch.Size([1, 12, 6, 6])
# 1 batch, 12 heads, 6 tokens, attending to 6 tokens

# Visualize first head of first layer
head_attention = attention[0][0, 0]  # First head
print(head_attention)
```

---

## 8. Batch Size

### What is Batch Size?
Number of samples processed together before updating model weights.  
- Larger batch = more stable gradients but more memory.  
- Smaller batch = noisier gradients but less memory.

### Batch Size Trade-offs

| Batch Size | Gradient Stability | Memory | Speed | Training Time |
|-----------|-------------------|--------|-------|----------------|
| 1 | Low ❌ | Low ✅ | Slow | Long ❌ |
| 32 | Medium ✓ | Medium ✓ | Fast ✅ | Medium ✓ |
| 128 | High ✅ | High ❌ | Fastest | Short ✅ |
| 256+ | Very High | Very High ❌ | - | - |

### Code Example
```python
from transformers import TrainingArguments, Trainer

# Small batch size (GPU constrained)
training_args = TrainingArguments(
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
)

# Large batch size (plenty of memory)
training_args = TrainingArguments(
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
)

# Gradient accumulation (simulate larger batch)
training_args = TrainingArguments(
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,  # Effective batch = 8 * 8 = 64
)
```

---

## 9. Learning Rate

### What is Learning Rate?
Controls how much model weights change during training.  
- Too low: Training too slow
- Too high: Training unstable, diverges

### Learning Rate Examples

**Typical Ranges:**
```
Fine-tuning: 1e-5 to 5e-5
Training from scratch: 1e-3 to 1e-4
```

**Effect:**

```
Learning Rate = 1e-3 (too high)
Loss: 2.5 → 5.0 → 10.0 ❌ Diverges

Learning Rate = 1e-4 (good)
Loss: 2.5 → 2.1 → 1.8 ✅ Converges smoothly

Learning Rate = 1e-6 (too low)
Loss: 2.5 → 2.48 → 2.47 (very slow)
```

### Learning Rate Schedules

```python
from transformers import get_linear_schedule_with_warmup

# Warmup then linear decay
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,
    num_training_steps=10000
)
# Learning rate: 0 → peak → 0 (linear decay)
```

### Code Example
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    learning_rate=2e-5,  # For fine-tuning
    warmup_steps=500,    # Gradual increase first
    num_train_epochs=3,
)
```

---

## 10. Max Tokens (Generation)

### What is Max Tokens?
Maximum length of generated output.  
- Prevents infinite generation.  
- Differs from context window (input limit).

### Code Example
```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompt = "The future of AI"

# Short response
output_short = generator(
    prompt,
    max_new_tokens=10,   # Generate only 10 new tokens
    do_sample=True
)

# Long response
output_long = generator(
    prompt,
    max_new_tokens=100,  # Generate up to 100 tokens
    do_sample=True
)

print("Short:", output_short[0]["generated_text"])
print("Long:", output_long[0]["generated_text"])
```

---

## Summary Table: All Parameters

| Concept | Purpose | Typical Values | Impact |
|---------|---------|-----------------|--------|
| **Tokenization** | Convert text to IDs | Vocab: 30K-100K | Affects model size |
| **Embedding Dim** | Vector representation size | 768-12,288 | Model capacity |
| **Context Window** | Max input length | 512-128K tokens | Memory needed |
| **Parameters** | Trainable weights | 110M-175B | Speed & accuracy |
| **Temperature** | Output randomness | 0-2 | Creativity |
| **Top-K** | Limit sampling | 50 | Quality control |
| **Top-P** | Nucleus sampling | 0.9 | Quality control |
| **Attention Heads** | Parallel mechanisms | 12-96 | Feature richness |
| **Batch Size** | Samples per step | 8-256 | Memory & speed |
| **Learning Rate** | Weight update step | 1e-6 to 1e-3 | Training stability |
| **Max Tokens** | Output limit | 50-2000 | Response length |

---

# 🚀 Product Indexer & Product Search - Setup & Usage

## 📋 Prerequisites

Ensure all required packages are installed:

```bash
pip install pandas joblib nltk spacy scikit-learn vaderSentiment gensim rapidfuzz
python -m spacy download en_core_web_sm
```

Download NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## 🔧 Product Indexer - Complete Guide

### What Does ProductIndexer Do?

The `ProductIndexer` class processes raw product data and creates:
- ✅ **TF-IDF vectors** for semantic search
- ✅ **Sentiment analysis** on product reviews  
- ✅ **Named Entity Recognition (NER)** for product attributes
- ✅ **Topic modeling** to discover product themes
- ✅ **ML classifier** for product categorization

### Step 1: Prepare Your Data

Create a JSON file with product data (one product per line):

**File: `electronics_tn.json`**
```json
{"product_name": "iPhone 15", "category": "Electronics", "brand": "Apple", "review_text": "Amazing phone, great camera", "price": 999, "seller_location": {"city": "New York", "lat": 40.7128, "lon": -74.0060}, "seller_rating": 4.8}
{"product_name": "Samsung Galaxy", "category": "Electronics", "brand": "Samsung", "review_text": "Good phone, fast processor", "price": 799, "seller_location": {"city": "Los Angeles", "lat": 34.0522, "lon": -118.2437}, "seller_rating": 4.5}
```

### Step 2: Run the Indexer

```python
from nlp.product_indexer import ProductIndexer

# Initialize and run the complete pipeline
indexer = ProductIndexer('nlp/electronics_tn.json')

# Output: Started → read_data → convert_data → combine_data 
#         → process_data → vectorize_data → sentiment_analysis 
#         → pos_ner_analysis → topic_modeling → train_classifier 
#         → store_data → Pipeline completed successfully.
```

### Step 3: Generated Output Files

| File | Location | Purpose |
|------|----------|---------|
| `products.parquet` | `nlp/` | Processed product data with all features |
| `vectorizer.pkl` | `nlp/` | TF-IDF vectorizer (transform new queries) |
| `tfidf_matrix.pkl` | `nlp/` | Pre-computed TF-IDF vectors |
| `classifier.pkl` | `nlp/` | Trained classification model |

### Pipeline Architecture

```
Input: electronics_tn.json
    ↓
1. read_data() → Load JSON into DataFrame
    ↓
2. convert_data() → Extract city, lat, lon from nested location object
    ↓
3. combine_data() → Merge all fields into product_text
    ↓
4. process_data() → Preprocess:
    - Lowercase
    - Remove punctuation
    - Remove stop words (except "not")
    - Lemmatize
    ↓
5. vectorize_data() → TF-IDF vectors (1-2 grams)
    ↓
6. pos_ner_analysis() → Extract Parts-of-Speech & Named Entities
    ↓
7. sentiment_analysis() → Analyze review sentiment (VADER)
    ↓
8. topic_modeling() → Discover topics using LDA
    ↓
9. train_classifier() → Train Logistic Regression classifier
    ↓
10. store_data() → Save .parquet, .pkl files
    ↓
Output: Ready for ProductSearch
```

---

## 🔍 Product Search - Complete Guide

### What Does ProductSearch Do?

The `ProductSearch` class loads indexed products and enables:
- ✅ Natural language query understanding
- ✅ Intelligent **price range detection** ("under 500", "between 1000-2000")
- ✅ **Multi-city filtering**
- ✅ **Semantic similarity ranking** using TF-IDF
- ✅ **Fuzzy matching** for typos

### Step 1: Initialize Search

```python
from nlp.product_search import ProductSearch

# Create search instance (loads pre-indexed data)
search = ProductSearch()

# Automatically loads:
# - products.parquet
# - vectorizer.pkl  
# - tfidf_matrix.pkl
```

### Step 2: Search Query Examples

#### Example 1: Simple Keyword Search
```python
results = search.search("laptop", top_n=10)

for idx, product in results.iterrows():
    print(f"{product['product_name']} - ${product['price']}")
```

#### Example 2: Price Range Search
```python
results = search.search("phone between 500 1000", top_n=5)

# Automatically detects:
# - Intent: "between"
# - Min price: 500
# - Max price: 1000
```

#### Example 3: Maximum Price Search
```python
results = search.search("laptop under 800", top_n=10)

# Detects:
# - Intent: "below/under"
# - Max price: 800
```

#### Example 4: Multi-City Filtering
```python
results = search.search("camera in New York Los Angeles", top_n=10)

# Filters by cities found in database
```

#### Example 5: Combined Search
```python
results = search.search("gaming laptop between 1000 2000 in New York", top_n=5)

# Combines: product + price range + cities
```

### Query Pattern Reference

```python
# Price patterns (auto-detected)
"below 500"               # Max price = 500
"under 1000"              # Max price = 1000
"less than 2000"          # Max price = 2000
"above 300"               # Min price = 300
"greater than 100"        # Min price = 100
"more than 500"           # Min price = 500
"between 500 1500"        # Min = 500, Max = 1500

# City filtering
"in New York"             # Single city
"in New York London"      # Multiple cities
"Chennai Bangalore"       # Multiple cities

# Combined
"phone under 500 in New York"
"laptop between 1000 2000 Chennai"
```

### Advanced Search Implementation

```python
from nlp.product_search import ProductSearch
from sklearn.metrics.pairwise import cosine_similarity

search = ProductSearch()

def advanced_search(query, top_n=10, min_score=0.1):
    # Preprocess and vectorize
    query_preprocessed = search.preprocess_text(query)
    query_vector = search.vectorizer.transform([query_preprocessed])
    
    # Compute similarities
    similarities = cosine_similarity(query_vector, search.tfidf_matrix)[0]
    
    # Filter by score
    valid_indices = similarities > min_score
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    results = search.data.iloc[top_indices].copy()
    results['similarity_score'] = similarities[top_indices]
    
    return results.sort_values('similarity_score', ascending=False)

# Usage
results = advanced_search("best smartphone camera", top_n=5)
print(results[['product_name', 'price', 'city', 'similarity_score']])
```

---

## 📱 Streamlit App - Complete Setup & Run

### What is simple.app.py?

Interactive web interface featuring:
- ✅ **English Grammar Correction** using GPT-3.5
- ✅ **Text Explanation** with examples
- ✅ **Image Generation** from prompts

### Installation

```bash
# Install dependencies
pip install streamlit openai python-dotenv

# Create .env file
cd streamlit
echo OPENAI_API_KEY=your_api_key_here > .env
```

### Configuration

**File: `streamlit/.env`**
```
OPENAI_API_KEY=sk-your-actual-key-here
```

Or set environment variable:
```powershell
# Windows PowerShell
$env:OPENAI_API_KEY="sk-your-key"

# Windows CMD
set OPENAI_API_KEY=sk-your-key
```

### Run the App

```bash
# Navigate to project root
cd C:\Users\rarju\Documents\ai\stepbystepAI

# Run streamlit
streamlit run streamlit/simple.app.py
```

Expected output:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501

  Press Ctrl+c to quit
```

### Using the App Features

#### Feature 1: Grammar Correction

1. Enter text with errors:
```
"The cat are sleeping on the bed and dont want to wake up."
```

2. Click **"Correct Grammar"**

3. Result:
```
"The cat is sleeping on the bed and doesn't want to wake up."
```

#### Feature 2: Text Explanation

1. Enter any topic:
```
"quantum computing"
```

2. Click **"Explain More"**

3. Receive detailed explanation with examples

#### Feature 3: Image Generation

1. Enter a prompt:
```
"a futuristic city with flying cars and neon lights"
```

2. Click **"Generate Image"**

3. Image appears in app and saves as `generated_image.png`

### Troubleshooting

| Problem | Solution |
|---------|----------|
| "API key not found" | Check `.env` file exists in `streamlit/` folder |
| "Port 8501 already in use" | `streamlit run streamlit/simple.app.py --server.port=8502` |
| "Model not found: gpt-3.5-turbo" | Verify OpenAI account has API access |
| Slow responses | Check internet connection and API rate limits |

---

## 🎯 Use Cases & Real-World Applications

### Use Case 1: E-Commerce Product Search

**Scenario:** Customer wants to find affordable smartphones in their city

**Query:** `"smartphone under 500 in Chennai"`

**What Happens:**
```
1. ProductSearch.search() is called
2. Intent detection: "below" (max price)
3. Price extraction: 500
4. City extraction: Chennai
5. Filter products by city
6. Filter by price (≤ 500)
7. Rank by TF-IDF similarity to "smartphone"
8. Return top 10 results
```

**Code Example:**
```python
from nlp.product_search import ProductSearch

search = ProductSearch()
results = search.search("smartphone under 500 in Chennai", top_n=10)

print(f"Found {len(results)} affordable smartphones in Chennai:")
for idx, row in results.iterrows():
    print(f"  • {row['product_name']} - ₹{row['price']} ({row['seller_rating']}★)")
```

### Use Case 2: Sentiment-Based Product Recommendations

**Scenario:** Find products with positive reviews

**Implementation:**
```python
from nlp.product_indexer import ProductIndexer
import pandas as pd

# Load indexed data
data = pd.read_parquet('nlp/products.parquet')

# Filter by positive sentiment
positive_products = data[data['sentiment_label'] == 'positive']
positive_products = positive_products.sort_values('sentiment_score', ascending=False)

print("Top-rated products by sentiment:")
print(positive_products[['product_name', 'sentiment_score', 'seller_rating']].head(10))
```

### Use Case 3: Multi-Criteria Smart Filter

**Scenario:** Customer needs: gaming laptop, high rating, good price, specific city

```python
from nlp.product_search import ProductSearch
import pandas as pd

search = ProductSearch()

# Get search results
results = search.search("gaming laptop", top_n=50)

# Apply multiple filters
filtered = results[
    (results['price'] <= 2000) &              # Price ≤ 2000
    (results['seller_rating'] >= 4.5) &      # Rating ≥ 4.5
    (results['city'].isin(['Chennai', 'Bangalore']))  # Specific cities
]

# Sort by relevance
filtered = filtered.sort_values('similarity_score', ascending=False)

print(f"\n🎮 Gaming Laptops matching all criteria: {len(filtered)} found")
for idx, row in filtered.head(5).iterrows():
    print(f"  {row['product_name']} - ₹{row['price']} ({row['seller_rating']}★) - {row['city']}")
```

### Use Case 4: Topic-Based Product Discovery

**Scenario:** Find all products discussing "battery life"

```python
from nlp.product_indexer import ProductIndexer
import pandas as pd

# Load indexed products
data = pd.read_parquet('nlp/products.parquet')

# Find products mentioning battery in topics
battery_products = data[data['topics'].str.contains('battery', na=False, case=False)]

print("Products discussing battery:")
print(battery_products[['product_name', 'review_text', 'topics']].head())
```

### Use Case 5: Named Entity Extraction for Specifications

**Scenario:** Extract all mentioned brands, models, features

```python
from nlp.product_indexer import ProductIndexer
import pandas as pd

data = pd.read_parquet('nlp/products.parquet')

# Look at extracted entities
product = data.iloc[0]

print(f"Product: {product['product_name']}")
print(f"\nExtracted Entities (NER):")
for entity, label in product['entities']:
    print(f"  • {entity} ({label})")

print(f"\nParts of Speech:")
for word, pos in product['pos_tags'][:10]:
    print(f"  • {word} ({pos})")
```

### Use Case 6: Batch Processing Multiple Queries

**Scenario:** Process multiple customer queries and log results

```python
from nlp.product_search import ProductSearch
import csv

search = ProductSearch()

queries = [
    "phone under 500",
    "laptop between 1000 2000 in Bangalore",
    "camera with good rating",
    "affordable tablets in Chennai"
]

results_log = []

for query in queries:
    results = search.search(query, top_n=3)
    
    for idx, product in results.iterrows():
        results_log.append({
            'query': query,
            'product': product['product_name'],
            'price': product['price'],
            'city': product['city'],
            'rating': product['seller_rating']
        })

# Save results
import pandas as pd
df = pd.DataFrame(results_log)
df.to_csv('search_results.csv', index=False)

print("✅ Search results saved to search_results.csv")
print(df.head(10))
```

---

## 🔗 Complete Integration Example

### Full Workflow: From Setup to Search

```bash
# Step 1: Install dependencies
pip install pandas joblib nltk spacy scikit-learn vaderSentiment gensim rapidfuzz streamlit openai python-dotenv

# Step 2: Download NLTK/spaCy models
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
python -m spacy download en_core_web_sm

# Step 3: Prepare data
# Place electronics_tn.json in nlp/ folder
```

### Python Script: Full Workflow

```python
#!/usr/bin/env python
"""
Complete workflow: Index products and search them
"""

from nlp.product_indexer import ProductIndexer
from nlp.product_search import ProductSearch
import pandas as pd

# ============================================
# STEP 1: INDEX PRODUCTS (Run once)
# ============================================
print("🔄 Step 1: Indexing products...")
try:
    indexer = ProductIndexer('nlp/electronics_tn.json')
    print("✅ Indexing complete!\n")
except Exception as e:
    print(f"❌ Indexing failed: {e}\n")

# ============================================
# STEP 2: INITIALIZE SEARCH
# ============================================
print("📂 Step 2: Loading indexed data...")
try:
    search = ProductSearch()
    print("✅ Search engine ready!\n")
except Exception as e:
    print(f"❌ Search initialization failed: {e}\n")
    exit()

# ============================================
# STEP 3: RUN SAMPLE QUERIES
# ============================================
print("🔍 Step 3: Running sample searches...\n")

sample_queries = [
    "phone",
    "laptop under 1000",
    "tablet between 300 600",
    "camera in New York"
]

for query in sample_queries:
    print(f"📌 Query: '{query}'")
    results = search.search(query, top_n=3)
    
    if len(results) > 0:
        for idx, row in results.iterrows():
            print(f"   • {row['product_name']} - ₹{row['price']} ({row['seller_rating']}★)")
    else:
        print("   ❌ No results found")
    print()

print("✅ Complete workflow finished!")
```

### Run the Complete Workflow

```bash
python nlp_workflow.py

# Expected Output:
# 🔄 Step 1: Indexing products...
# Started
# read_data
# convert_data
# ...
# Pipeline completed successfully.
# ✅ Indexing complete!
#
# 📂 Step 2: Loading indexed data...
# ✅ Search engine ready!
#
# 🔍 Step 3: Running sample searches...
#
# 📌 Query: 'phone'
#    • iPhone 15 - ₹999 (4.8★)
#    • Samsung Galaxy - ₹799 (4.5★)
# ...
```

---

## 📊 Performance & Optimization Tips

### For ProductIndexer
- **Test with small data first** (10-100 products) before large datasets
- **Monitor memory** for large databases (>100K products)
- **Cache vectorizer** after first run to avoid recomputation
- **Parallel processing**: Use `n_jobs=-1` in sklearn functions

### For ProductSearch
- **Pre-compute vectors** during indexing (already done)
- **Cache frequent queries** to avoid re-vectorization
- **Use similarity thresholds** to filter low-relevance results
- **Batch process** multiple queries together

### For Streamlit App
- **Load models once**: Use `@st.cache_resource`
- **Cache data**: Use `@st.cache_data` for API responses
- **Set timeouts** for OpenAI API calls (default 30s)
- **Monitor API costs** - each request incurs charges

### Memory Usage Example

```python
import psutil
import os

def check_memory():
    process = psutil.Process(os.getpid())
    info = process.memory_info()
    print(f"Memory used: {info.rss / 1024 / 1024:.2f} MB")

# Before indexing
check_memory()

# After indexing (may increase significantly)
check_memory()
```

---

## 🐛 Debugging Common Issues

### Issue: "products.parquet not found"
```python
# Ensure ProductIndexer has completed successfully
# Check if files exist:
import os
print(os.path.exists('nlp/products.parquet'))
print(os.path.exists('nlp/vectorizer.pkl'))
print(os.path.exists('nlp/tfidf_matrix.pkl'))
```

### Issue: "No results found"
```python
# Check data is not empty
data = pd.read_parquet('nlp/products.parquet')
print(f"Total products: {len(data)}")
print(f"Sample products: {data['product_name'].head()}")

# Verify similarity calculation
query = "laptop"
preprocessed = search.preprocess_text(query)
print(f"Preprocessed query: {preprocessed}")
```

### Issue: "Memory error during indexing"
```python
# Process in batches
import pandas as pd

# Process large JSON in chunks
chunks = pd.read_json('file.json', lines=True, chunksize=1000)
for chunk in chunks:
    # Process chunk
    pass
```
