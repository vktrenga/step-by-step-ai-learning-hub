# 🌐 LangChain Deep Dive

## 1. What is LangChain?
- **Definition**: A framework for building applications powered by Large Language Models (LLMs).  
- **Purpose**: It connects models (like OpenAI GPT), prompts, memory, and external tools into structured workflows.  
- **Use cases**:
  - Chatbots (customer support, travel assistant)
  - Document Q&A (searching PDFs, contracts)
  - Workflow automation (summarization, translation pipelines)
  - Retrieval-Augmented Generation (RAG) with custom data

---

## 2. OpenAI API Integration
- LangChain wraps **OpenAI APIs** (`gpt-4`, `gpt-3.5`) for text/chat tasks.  
- Provides structured access to:
  - **System messages** (define role/behavior)
  - **User messages** (human input)
  - **Assistant messages** (model output)

**Example Use Case**:  
A chatbot where the system message is *“You are a helpful travel assistant”*, the user asks *“Find me flights to Paris”*, and the assistant responds with flight details.

---

## 3. Message Types
- **System message**: Defines role/behavior.  
  Example: `"You are a financial advisor."`
- **Human/User message**: Input from the user.  
  Example: `"Suggest investment options for 2026."`
- **AI/Assistant message**: Model’s response.  
  Example: `"Consider diversified ETFs and bonds."`

---

## 4. Model Parameters
- **Temperature**: Controls creativity.  
  - `0` → deterministic (good for coding, facts)  
  - `1` → creative (good for storytelling)  
- **Max tokens**: Response length limit.  
- **Streaming**: Outputs tokens live (useful for chat apps).  
- **Seed**: Ensures reproducibility of outputs.

**Use Case**:  
- Customer support bot → `temperature=0`  
- Creative writing assistant → `temperature=0.9`

---

## 5. Prompt Engineering
### Prompt Templates
- Reusable text with placeholders.  
```python
from langchain.prompts import PromptTemplate
template = PromptTemplate.from_template("Translate {text} to French")
```

### Chat Prompt Templates
- Multi-message conversations.  
```python
from langchain.prompts import ChatPromptTemplate
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a travel assistant."),
    ("human", "{question}")
])
```

### Few-Shot Prompts
- Add examples to guide responses.  
```python
examples = [
    ("human", "Translate 'Hello' to French"),
    ("ai", "Bonjour")
]
```

**Use Case**:  
Few-shot prompts help train the model to follow a specific style (e.g., polite customer replies).

---

## 6. Output Parsers
- **StringOutputParser**: Returns plain text.  
- **CommaSeparatedListOutputParser**: Splits into list.  
- **DatetimeOutputParser**: Parses dates/times.

**Example**:  
- Input: `"Paris, Rome, London"`  
- Parser: `CommaSeparatedListOutputParser`  
- Output: `["Paris", "Rome", "London"]`

---

## 7. Execution Concepts

### Piping (Prompt → Model → Parser)
**What**: Chains multiple runnables together using the `|` operator.  
**Why**: Creates a clean, readable pipeline where output of one step feeds into the next.  
**How**: Each component must be a `Runnable` with `.invoke()` method.

**Basic Example:**
```python
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import StrOutputParser

# Define individual components
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}")
])

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
parser = StrOutputParser()

# Chain together using pipe operator
chain = prompt | model | parser

# Execute the chain
result = chain.invoke({"question": "Explain machine learning in 2 sentences."})
print(result)
# Output: "Machine learning is a subset of AI where systems learn patterns from data..."
```

**Advanced Example with Data Transformation:**
```python
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

# Custom transformation function
def extract_entities(text):
    return {"entities": text.split(","), "length": len(text.split(","))}

# Chain with custom lambda
complex_chain = (
    prompt 
    | model 
    | parser 
    | RunnableLambda(extract_entities)
)

result = complex_chain.invoke({"question": "List fruits: apple, banana, orange"})
# Output: {"entities": ["apple", "banana", "orange"], "length": 3}
```

**Use Cases**:
- Simple Q&A workflows
- Document summarization pipelines
- Multi-step data transformations
- Real-time processing chains

---

### Batching
**What**: Process multiple inputs at once using `.batch()` instead of `.invoke()`.  
**Why**: More efficient than calling `.invoke()` in a loop; leverages parallel processing.  
**How**: Pass a list of dictionaries; outputs a list of results.

**Example 1: Translate Multiple Descriptions**
```python
# Define chain (same as before)
chain = prompt | model | parser

# Batch input
product_descriptions = [
    {"question": "Translate to French: The laptop is powerful."},
    {"question": "Translate to French: The phone is affordable."},
    {"question": "Translate to French: The tablet is portable."}
]

# Execute batch
results = chain.batch(product_descriptions)
print(results)
# Output: ["L'ordinateur portable est puissant.", "Le téléphone est abordable.", "La tablette est portable."]
```

**Example 2: Batch Sentiment Analysis**
```python
from langchain.prompts import PromptTemplate

sentiment_prompt = PromptTemplate.from_template(
    "Analyze the sentiment of this review: {review}. Reply with: positive, negative, or neutral."
)

sentiment_chain = sentiment_prompt | model | parser

reviews = [
    {"review": "This product is amazing! Highly recommend."},
    {"review": "Terrible quality, would not buy again."},
    {"review": "It's okay, nothing special."}
]

sentiments = sentiment_chain.batch(reviews)
print(sentiments)
# Output: ["positive", "negative", "neutral"]
```

**Performance Comparison:**
```python
import time

# ❌ Inefficient: Loop with invoke
start = time.time()
results_loop = []
for item in product_descriptions:
    result = chain.invoke(item)
    results_loop.append(result)
time_loop = time.time() - start
print(f"Loop time: {time_loop:.2f}s")

# ✅ Efficient: Batch
start = time.time()
results_batch = chain.batch(product_descriptions)
time_batch = time.time() - start
print(f"Batch time: {time_batch:.2f}s")
# Batch is typically 2-3x faster
```

**Use Cases**:
- Processing bulk data (100+ items)
- Parallel API calls
- Batch processing in data pipelines
- Performance-critical workflows

---

### Streaming
**What**: Outputs tokens/chunks in real-time instead of waiting for complete response.  
**Why**: Better UX for chat apps; can display results as they arrive.  
**How**: Use `.stream()` instead of `.invoke()`; iterate over chunks.

**Example 1: Basic Streaming to Console**
```python
# Same chain setup
chain = prompt | model | parser

# Stream output
print("Streaming response: ", end="", flush=True)
for chunk in chain.stream({"question": "Tell a short story about a robot."}):
    print(chunk, end="", flush=True)
print()
# Output: Streaming response: Once upon a time, a robot named... (appears word by word)
```

**Example 2: Streaming to Web Interface**
```python
from flask import Flask, Response

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat_stream():
    question = request.json.get('question')
    
    def generate():
        for chunk in chain.stream({"question": question}):
            yield chunk + "\n"
    
    return Response(generate(), mimetype='text/plain')
# Client receives: "Once" → "upon" → "a" → "time" (progressive updates)
```

**Example 3: Streaming with Aggregation**
```python
# Stream and collect chunks
full_response = ""
for chunk in chain.stream({"question": "Explain quantum computing."}):
    full_response += chunk
    # Optional: Do something with partial response
    if len(full_response) % 50 == 0:
        print(f"[Progress: {len(full_response)} chars received]")

print("Final response:", full_response)
```

**Use Cases**:
- Chatbot interfaces (Discord, Slack, web apps)
- Real-time data analysis dashboards
- Live translation or summarization
- Long-form content generation (stories, articles)
- Improving perceived responsiveness

---

### Combining All Three Concepts

**Real-World Travel Assistant Example:**
```python
from langchain.schema.runnable import RunnableParallel

# Phase 1: Piping - Structure the request
travel_chain = (
    prompt 
    | model 
    | parser
)

# Phase 2: Batching - Process multiple trip requests
trip_requests = [
    {"question": "Plan 5-day trip to Paris."},
    {"question": "Plan 7-day trip to Tokyo."},
    {"question": "Plan 3-day trip to Barcelona."}
]

# Get structured itineraries in parallel
itineraries = travel_chain.batch(trip_requests, config={"max_concurrency": 3})

# Phase 3: Streaming - Display results to user in real-time
for i, itinerary in enumerate(itineraries, 1):
    print(f"\n--- Trip {i} ---")
    # Simulate streaming (in real app, would stream from LLM)
    for chunk in itinerary.split(". "):
        print(chunk, end=". ", flush=True)
        time.sleep(0.5)  # Simulate network delay
    print()
```

---

### Comparison Table

| Concept | Method | Use Case | Speed | Streaming |
|---------|--------|----------|-------|-----------|
| **Piping** | `.invoke()` | Single request, Q&A | Moderate | No |
| **Batching** | `.batch()` | Multiple requests, bulk processing | Fast | No |
| **Streaming** | `.stream()` | Real-time chat, live output | Moderate | Yes |
| **Combined** | `.batch()` + `.stream()` | Bulk with real-time display | Fast | Yes |

**When to Use Each:**
- **Piping alone**: Quick test, single user query
- **Batching alone**: Data processing, back-office jobs
- **Streaming alone**: Chat app, interactive interface
- **All three**: Production travel assistant, enterprise chatbot

---

# ⚙️ Runnable Abstractions in LangChain

## 1. **Runnable**
- **What**: The base interface for anything executable in LangChain.  
- **Where**: Used for models, prompts, parsers, or chains.  
- **How**: Provides methods like `.invoke()`, `.batch()`, `.stream()`.  

**Example:**
```python
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo")
print(llm.invoke("Explain supply chain challenges"))
```
👉 Here, the model itself is a **Runnable**.

**Use Case**:  
- Directly call an LLM for Q&A.  
- Run a parser to clean raw text.

---

## 2. **RunnableSequence**
- **What**: Connects multiple runnables into a pipeline.  
- **Where**: When you want step-by-step processing.  
- **How**: Each step’s output becomes the next step’s input.  

**Example:**
```python
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
from langchain.chat_models import ChatOpenAI

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a manufacturing consultant."),
    ("human", "{question}")
])

llm = ChatOpenAI(model="gpt-3.5-turbo")
parser = StrOutputParser()

chain = RunnableSequence(first=prompt, middle=[llm], last=parser)
print(chain.invoke({"question": "List key challenges in production scheduling"}))
```

**Use Case**:  
- **Manufacturing**: Convert messy notes → structured schedule.  
- **Retail**: Customer query → clean structured answer.

---

## 3. **Piping Chains (`|`)**
- **What**: Shorthand for chaining runnables.  
- **Where**: Cleaner syntax for sequences.  
- **How**: `prompt | model | parser`.

**Example:**
```python
chain = prompt | llm | parser
print(chain.invoke({"trip": "want trip Paris Rome July 2026"}))
```

**Use Case**:  
- **Travel assistant**: Convert rough trip request → structured itinerary.

---

## 4. **RunnablePassthrough**
- **What**: Passes input unchanged.  
- **Where**: Useful for logging, debugging, or keeping raw input.  
- **How**: Keeps original input alongside processed output.

**Example:**
```python
from langchain.schema.runnable import RunnablePassthrough

passthrough = RunnablePassthrough()
print(passthrough.invoke({"raw_input": "buy shoes deliver tomorrow"}))
```

**Use Case**:  
- **Retail**: Keep original messy order text while also parsing structured data.  
- **Debugging**: Compare raw vs processed inputs.

---

## 5. **Grouping Runnable**
- **What**: Run multiple runnables in parallel and combine outputs.  
- **Where**: When you need multiple perspectives from the same input.  
- **How**: `RunnableParallel`.

**Example:**
```python
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.schema.runnable import RunnableParallel

list_parser = CommaSeparatedListOutputParser()

grouped_chain = RunnableParallel({
    "text": prompt | llm | parser,
    "list": prompt | llm | list_parser
})

result = grouped_chain.invoke({"trip": "buy shoes, laptop deliver 2026-06-10"})
print(result)
```

**Output:**
```python
{
  "text": "I want to buy shoes and a laptop, delivered on 2026-06-10.",
  "list": ["shoes", "laptop"]
}
```

**Use Case**:  
- **Retail order assistant**: One branch → corrected text, another → product list.  
- **Travel assistant**: One branch → itinerary text, another → destination list.

---

## 6. **RunnableParallel**
- **What**: Explicit abstraction for parallel execution.  
- **Where**: When tasks don’t depend on each other.  
- **How**: Each branch runs independently, results merged.

**Example:**
```python
parallel_chain = RunnableParallel({
    "flights": prompt | llm | parser,
    "hotels": prompt | llm | parser,
    "weather": prompt | llm | parser
})

result = parallel_chain.invoke({"trip": "Plan trip to Paris"})
```

**Use Case**:  
- **Travel assistant**: Fetch flights, hotels, weather simultaneously.  
- **Business intelligence**: Summarize multiple reports in parallel.

---

# ✅ Key Takeaways
- **Runnable** = basic unit.  
- **RunnableSequence** = ordered pipeline.  
- **Piping (`|`)** = shorthand chaining.  
- **RunnablePassthrough** = keep raw input.  
- **Grouping Runnable** = multiple outputs from same input.  
- **RunnableParallel** = independent tasks in parallel.  

---



# 🧠 Memory & Conversation History

## ConversationBufferMemory
Stores all conversation history without summarization.

**Example:**
```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI

memory = ConversationBufferMemory()
llm = ChatOpenAI(model="gpt-3.5-turbo")

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# First interaction
response1 = conversation.run(input="My name is Alice. I like Python.")
# LLM remembers: Alice likes Python

# Second interaction
response2 = conversation.run(input="What do I like?")
# Output: "You like Python"  (recalls from memory)
```

**Use Case**: Simple chatbots, customer support where full history matters.

---

## ConversationSummaryMemory
Summarizes conversation to save tokens.

**Example:**
```python
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(llm=llm)

# Long conversation gets summarized
# Instead of storing 1000 tokens of history, stores 100-token summary
```

**Use Case**: Long-running conversations, cost optimization.

---

## ConversationBufferWindowMemory
Keeps only last N messages (sliding window).

**Example:**
```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=5)  # Keep last 5 messages
conversation = ConversationChain(llm=llm, memory=memory)
```

**Use Case**: Prevent context explosion, focus on recent conversation.

---

# 🔍 Retrievers & Vector Stores (RAG)

## What is RAG (Retrieval-Augmented Generation)?
Combines **retrieval** (finding relevant documents) with **generation** (answering questions).

**Flow:**
```
Question → Retrieve relevant documents → Inject into prompt → Generate answer
```

---

## Vector Store Example
```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

# Step 1: Load documents
loader = PyPDFLoader("manual.pdf")
docs = loader.load()

# Step 2: Split into chunks
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunks = splitter.split_documents(docs)

# Step 3: Create embeddings
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(chunks, embeddings)

# Step 4: Search
query = "How do I reset the device?"
relevant_docs = vector_store.similarity_search(query, k=3)
# Returns 3 most relevant chunks
```

---

## RetrievalQA (Question Answering with Documents)
```python
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Simple concatenation
    retriever=vector_store.as_retriever(k=3)
)

answer = qa_chain.run("How do I reset the device?")
# Output: Answers based on PDF content, not general knowledge
```

**Use Case**: Customer support, internal document Q&A, knowledge base search.

---


# 🤖 Agents & Tools

## What are Agents?
Agents use **reasoning** to decide which **tools** to call.

**Flow:**
```
Question → Agent thinks → Decides which tool → Calls tool → Gets result → Decides next step → Final answer
```

---

## Agent with Tools
```python
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun

# Define tools
search = DuckDuckGoSearchRun()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for answering questions about current events"
    ),
    Tool(
        name="Calculator",
        func=lambda x: str(eval(x)),
        description="Useful for math calculations"
    )
]

# Create agent
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

# Ask question
response = agent.run("What's 5 + 3? And search for latest AI news.")
# Agent decides:
# 1. Call Calculator for 5+3
# 2. Call Search for AI news
```

**Use Case**: Autonomous workflows, multi-step reasoning tasks.

---

## Custom Tool Example
```python
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get weather for a location"""
    # Call weather API
    return f"Weather in {location}: Sunny, 72°F"

@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert between currencies"""
    # Call currency API
    return f"{amount} {from_currency} = {amount * 1.1} {to_currency}"

# Use in agent
tools = [get_weather, convert_currency]
agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)

response = agent.run("What's the weather in Paris? Convert 100 USD to EUR.")
```

---

# ⛓️ Chains (Legacy but Important)

## LLMChain (Simple Pipeline)
```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

prompt = PromptTemplate.from_template("Translate to French: {text}")
llm = ChatOpenAI(model="gpt-3.5-turbo")

chain = LLMChain(prompt=prompt, llm=llm)
result = chain.run(text="Hello world")
# Output: "Bonjour le monde"
```

---

## SequentialChain (Multiple Steps)
```python
from langchain.chains import SequentialChain

# Step 1: Generate plot
plot_prompt = PromptTemplate.from_template("Write a plot for: {topic}")
plot_chain = LLMChain(llm=llm, prompt=plot_prompt, output_key="plot")

# Step 2: Write screenplay
screenplay_prompt = PromptTemplate.from_template("Write screenplay for: {plot}")
screenplay_chain = LLMChain(llm=llm, prompt=screenplay_prompt, output_key="screenplay")

# Combine
overall_chain = SequentialChain(
    chains=[plot_chain, screenplay_chain],
    input_variables=["topic"],
    output_variables=["plot", "screenplay"]
)

result = overall_chain({"topic": "A robot learns to love"})
# Output: {"plot": "...", "screenplay": "..."}
```

---

# 📞 Callbacks (Monitoring & Logging)

## Custom Callback
```python
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI

class MyCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"🚀 LLM Starting. Prompt: {prompts[0][:50]}...")
    
    def on_llm_end(self, response, **kwargs):
        print(f"✅ LLM Finished. Output: {response.generations[0][0].text[:50]}...")
    
    def on_chain_error(self, error, **kwargs):
        print(f"❌ Error: {error}")

# Use callback
llm = ChatOpenAI(callbacks=[MyCallback()])
response = llm.predict("Hello!")
# Output: 🚀 LLM Starting... ✅ LLM Finished...
```

**Use Case**: Debugging, performance monitoring, cost tracking, logging.

---

# ⚠️ Error Handling

## Try-Catch with Callbacks
```python
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chat_models import ChatOpenAI

try:
    llm = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=5)
    response = llm.predict("Write a very long story...")
except Exception as e:
    print(f"Error: {e}")
    # Handle gracefully
```

## Retry Logic
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def call_llm(prompt):
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    return llm.predict(prompt)

# Retries up to 3 times with exponential backoff
```

---

# 🔗 Complete RAG Pipeline

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# Step 1: Load & process documents
loader = PyPDFLoader("company_handbook.pdf")
documents = loader.load()
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Step 2: Create vector store
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(chunks, embeddings)

# Step 3: Create QA chain with memory
memory = ConversationBufferMemory()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(k=3),
    memory=memory
)

# Step 4: Ask questions
q1 = qa_chain.run("What's the company's vacation policy?")
q2 = qa_chain.run("How many days did I ask about earlier?")  # Remembers q1
```

---

# 📊 LangChain Ecosystem Summary

| Component | Purpose | Example |
|-----------|---------|---------|
| **Prompts** | Structure input | `PromptTemplate`, `ChatPromptTemplate` |
| **LLMs** | Call models | `ChatOpenAI`, `HuggingFaceLLM` |
| **Memory** | Store history | `BufferMemory`, `SummaryMemory` |
| **Retrievers** | Find documents | `FAISS`, `Pinecone`, `Weaviate` |
| **Tools** | External functions | `Search`, `Calculator`, custom tools |
| **Agents** | Reasoning + tools | `ReActAgent`, `OpenAI Functions` |
| **Chains** | Workflow pipelines | `LLMChain`, `SequentialChain`, `RAG` |
| **Callbacks** | Monitoring | Custom logging, cost tracking |
| **Loaders** | Ingest data | `PyPDFLoader`, `WebBaseLoader`, `CSVLoader` |

---

