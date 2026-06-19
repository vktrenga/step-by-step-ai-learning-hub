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

Prompt engineering is the art of crafting inputs to get desired outputs from LLMs. Different prompting techniques work better for different tasks.

---

### 5.1 Zero-Shot Prompting
**What**: Ask the model to perform a task without any examples.  
**When to use**: General knowledge questions, straightforward tasks.  
**Pros**: Quick, no examples needed.  
**Cons**: May lack specificity; unpredictable outputs.

**Example:**
```python
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import StrOutputParser

# Zero-shot: No examples, just direct instruction
prompt = PromptTemplate.from_template(
    "Classify the sentiment of this review: {review}\n"
    "Answer with only: positive, negative, or neutral."
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
chain = prompt | llm | StrOutputParser()

result = chain.invoke({"review": "This product is amazing! Worth every penny."})
print(result)  # Output: positive
```

**Use Cases**:
- Answering factual questions
- Simple classifications
- Creative tasks
- Brainstorming

---

### 5.2 One-Shot Prompting
**What**: Provide a single example to guide the model's behavior.  
**When to use**: When you need slightly more control than zero-shot.  
**Pros**: Adds specificity with minimal overhead.  
**Cons**: One example may not be enough for complex tasks.

**Example:**
```python
from langchain.prompts import PromptTemplate

# One-shot: Single example provided
one_shot_prompt = PromptTemplate.from_template(
    "Extract person's name and age from text.\n\n"
    "Example:\n"
    "Input: 'John is 25 years old and works as an engineer.'\n"
    "Output: Name: John, Age: 25\n\n"
    "Now do the same for:\n"
    "Input: {text}\n"
    "Output:"
)

result = chain.invoke({
    "text": "Sarah is 32 years old and is a doctor."
})
print(result)  # Output: Name: Sarah, Age: 32
```

**Use Cases**:
- Format/style imitation
- Pattern demonstration
- Simple rule following
- Initial prototyping

---

### 5.3 Few-Shot Prompting
**What**: Provide multiple examples (typically 2-5) to establish a pattern.  
**When to use**: Complex tasks requiring pattern recognition.  
**Pros**: More reliable; teaches the model your specific style.  
**Cons**: Uses more tokens; requires good example selection.

**Example 1: Sentiment Analysis with Few-Shot**
```python
from langchain.prompts import PromptTemplate

few_shot_prompt = PromptTemplate.from_template(
    "Classify sentiment and provide confidence score (0-100).\n\n"
    "Examples:\n"
    "1. Review: 'Excellent product, fast shipping!'\n"
    "   Sentiment: positive | Confidence: 95\n\n"
    "2. Review: 'Terrible quality, broke after 2 days.'\n"
    "   Sentiment: negative | Confidence: 98\n\n"
    "3. Review: 'It works, nothing special.'\n"
    "   Sentiment: neutral | Confidence: 85\n\n"
    "Now classify:\n"
    "Review: {review}\n"
    "Sentiment:"
)

result = chain.invoke({
    "review": "Great value for money, very happy with purchase."
})
print(result)  # Output: positive | Confidence: 92
```

**Example 2: Few-Shot with Few-Shot Prompt Template**
```python
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts import PromptTemplate

# Define examples
examples = [
    {
        "question": "What is the capital of France?",
        "answer": "Paris"
    },
    {
        "question": "What is the capital of Japan?",
        "answer": "Tokyo"
    },
    {
        "question": "What is the capital of Brazil?",
        "answer": "Brasília"
    }
]

# Define how each example should be formatted
example_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="Q: {question}\nA: {answer}"
)

# Create few-shot template
few_shot_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Q: {input}\nA:",
    input_variables=["input"]
)

# Use it
formatted_prompt = few_shot_template.format(input="What is the capital of India?")
print(formatted_prompt)
# Outputs: Q: What is the capital of France?\nA: Paris\n...Q: What is the capital of India?\nA:
```

**Use Cases**:
- Entity extraction
- Structured output formatting
- Task-specific reasoning
- Domain-specific knowledge tasks

---

### 5.4 Chain-of-Thought (CoT) Prompting
**What**: Ask the model to explain its reasoning step-by-step.  
**When to use**: Complex reasoning, math problems, multi-step decisions.  
**Pros**: Significantly improves accuracy; shows reasoning; easier to debug.  
**Cons**: Uses more tokens; slower responses.

**Example:**
```python
# Without Chain-of-Thought
simple_prompt = PromptTemplate.from_template(
    "Solve: {problem}"
)

# With Chain-of-Thought
cot_prompt = PromptTemplate.from_template(
    "Solve this step by step.\n\n"
    "Problem: {problem}\n\n"
    "Step 1: Understand the problem\n"
    "Step 2: Break it into smaller parts\n"
    "Step 3: Solve each part\n"
    "Step 4: Combine the solutions\n\n"
    "Solution:"
)

# Example execution
result = chain.invoke({
    "problem": "If Alice has 3 apples and Bob gives her 2 more, "
               "then Alice gives 1/2 of her apples to Charlie, "
               "how many apples does Alice have?"
})
# Model will show: Step 1: Alice starts with 3..., Step 2: Bob gives 2...
```

**Use Cases**:
- Mathematical reasoning
- Logical deduction
- Decision-making processes
- Technical problem-solving
- Multi-step workflows

---

### 5.5 Self-Consistency Prompting
**What**: Generate multiple Chain-of-Thought responses and select the most common answer.  
**When to use**: High-stakes decisions, critical accuracy needed.  
**Pros**: Significantly higher accuracy (studies show +5-10% improvement).  
**Cons**: N times more expensive (calls model N times).

**Example:**
```python
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from collections import Counter

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)  # High temp for diversity

cot_prompt = PromptTemplate.from_template(
    "Solve step by step: {problem}\n"
    "Final answer:"
)

# Generate multiple solutions (self-consistency)
num_solutions = 5
results = []

for i in range(num_solutions):
    result = chain.invoke({"problem": "What is 15% of 240?"})
    # Extract final number from result
    final_answer = result.split("Final answer:")[-1].strip()
    results.append(final_answer)

# Get most common answer (majority voting)
most_common = Counter(results).most_common(1)[0][0]
print(f"Most consistent answer: {most_common}")
# Output: 36
```

**Use Cases**:
- Medical diagnosis support
- Financial decision-making
- Critical bug fixing
- Risk assessment

---

### 5.6 Role-Based / Persona Prompting
**What**: Give the model a specific role or persona to adopt.  
**When to use**: When you need consistent style, tone, or expertise level.  
**Pros**: Better control over response style and expertise.  
**Cons**: Model may stay in character even when it shouldn't.

**Example:**
```python
from langchain.prompts import ChatPromptTemplate

# Define persona in system message
role_based_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert software architect with 15 years of experience. "
        "You provide clear, concise architectural decisions with trade-offs. "
        "Use technical terms but explain them. Format your response with: "
        "RECOMMENDATION, PROS, CONS, IMPLEMENTATION STEPS."
    ),
    ("human", "{question}")
])

result = chain.invoke({
    "question": "Should we use microservices or monolithic architecture?"
})
# Response will be in the voice of a senior architect, with structured format
```

**Use Cases**:
- Customer support (different personas for different departments)
- Creative writing (author, poet, journalist)
- Domain-specific advice (lawyer, doctor, engineer)
- Training/educational content

---

### 5.7 Instruction-Based Prompting
**What**: Give explicit, detailed instructions about what to do and what NOT to do.  
**When to use**: When you need tight control over model behavior.  
**Pros**: Most control; clear expectations; prevents undesired behaviors.  
**Cons**: Requires more careful prompt engineering.

**Example:**
```python
from langchain.prompts import PromptTemplate

instruction_prompt = PromptTemplate.from_template(
    "You are a product review summarizer.\n\n"
    "INSTRUCTIONS:\n"
    "1. Extract key points from the review (max 3 points)\n"
    "2. Rate sentiment: positive/negative/neutral\n"
    "3. Keep summary under 50 words\n"
    "4. Do NOT add opinions or assumptions\n"
    "5. Do NOT change the reviewer's words\n"
    "6. Do NOT rate the product quality separately from the review\n\n"
    "Review: {review}\n\n"
    "Summary:"
)

result = chain.invoke({
    "review": "This laptop is fast and the screen is beautiful, "
              "but the battery only lasts 3 hours which is disappointing."
})
```

**Use Cases**:
- Content moderation
- Data extraction
- Quality assurance
- Compliance checking

---

### 5.8 ReAct (Reasoning + Acting) Prompting
**What**: Combine reasoning with tool usage in a loop (Thought → Action → Observation).  
**When to use**: Complex tasks requiring external information or actions.  
**Pros**: Combines reasoning power with real-world actions; transparent thought process.  
**Cons**: More complex; multiple tool calls needed.

**Example:**
```python
from langchain.prompts import PromptTemplate

react_prompt = PromptTemplate.from_template(
    "Answer the question using this format:\n"
    "Thought: Do I need to search for this?\n"
    "Action: If yes, use Search tool\n"
    "Observation: What did I find?\n"
    "Thought: Do I have enough info?\n"
    "Final Answer: {question}\n\n"
    "Question: {question}"
)

# This would be used with an agent that has tools like Search, Calculator
# The agent would loop through Thought → Action → Observation until it has an answer
```

**Use Cases**:
- Information retrieval from web
- Complex calculations with lookups
- Real-time data queries
- Multi-step planning

---

### 5.9 Least-to-Most Prompting
**What**: Solve simpler sub-problems first, then use those solutions for harder ones.  
**When to use**: Complex problems that break into smaller pieces.  
**Pros**: Better accuracy on hard problems; builds solutions progressively.  
**Cons**: Requires problem decomposition; more tokens.

**Example:**
```python
from langchain.prompts import PromptTemplate

# Step 1: Solve simple version
simple_prompt = PromptTemplate.from_template(
    "Simplify this: {problem}\n"
    "Simpler version:"
)

# Step 2: Solve simple version
simple_result = simple_chain.invoke({"problem": "Calculate compound interest over 10 years with 5% rate on $1000"})
# Result: "How much is $1000 with 5% added 10 times?"

# Step 3: Use simple solution to solve hard version
complex_prompt = PromptTemplate.from_template(
    "Using this simpler understanding: {simpler}\n"
    "Now solve the original: {problem}"
)

final_result = chain.invoke({
    "simpler": simple_result,
    "problem": "Calculate compound interest over 10 years with 5% rate on $1000"
})
```

**Use Cases**:
- Scientific problem-solving
- Algorithm design
- Complex business logic
- Tutoring systems

---

### 5.10 Template Types in LangChain

#### Prompt Templates
- Reusable text with placeholders.  
```python
from langchain.prompts import PromptTemplate
template = PromptTemplate.from_template("Translate {text} to French")
prompt = template.format(text="Hello world")
```

#### Chat Prompt Templates
- Multi-message conversations with role distinction.  
```python
from langchain.prompts import ChatPromptTemplate
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a travel assistant."),
    ("human", "{question}")
])
messages = chat_template.format_messages(question="Plan my trip")
```

#### Few-Shot Prompt Template
- Structured examples with formatting.  
```python
from langchain.prompts.few_shot import FewShotPromptTemplate

examples = [
    {"input": "happy", "output": "😊"},
    {"input": "sad", "output": "😢"}
]

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Word: {input} → Emoji: {output}"
)

few_shot = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Word: {word} → Emoji:",
    input_variables=["word"]
)

result = few_shot.format(word="angry")
```

---

### 5.11 Prompt Engineering Best Practices

| Technique | Best For | Token Cost | Accuracy |
|-----------|----------|-----------|----------|
| **Zero-Shot** | General questions, simple tasks | Low | Baseline |
| **One-Shot** | Style imitation | Low-Medium | +10% |
| **Few-Shot** | Pattern learning | Medium | +20-30% |
| **Chain-of-Thought** | Reasoning tasks | Medium-High | +15-25% |
| **Self-Consistency** | Critical decisions | Very High | +5-10% |
| **Role-Based** | Style consistency | Low | Better quality |
| **Instruction-Based** | Controlled output | Low-Medium | +10-20% |
| **ReAct** | Complex workflows | High | +30-40% |
| **Least-to-Most** | Hard problems | High | +20-35% |

**General Tips:**
- Start with zero-shot; add examples only if needed
- Use specific, clear language
- Include format examples in the prompt
- Test different temperatures (0 for facts, 0.7+ for creativity)
- Put examples BEFORE the request (in context)
- Be explicit about length, format, and constraints
- Iterate and measure: track which prompts work best

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

### Core Definition
RAG is a hybrid approach that combines **information retrieval** with **generative AI**. Instead of relying solely on an LLM's pre-trained knowledge, RAG:
1. **Retrieves** relevant documents from a knowledge base using semantic search
2. **Augments** the LLM prompt with retrieved context
3. **Generates** answers grounded in real, updated information

### How RAG Works (Detailed)

**Basic Flow:**
```
User Query → Embed Query → Search Vector Store → Retrieve Top-K Documents → 
Add to Prompt Context → LLM Generates Answer → Return Answer
```

**Detailed Process:**
```
1. User Input: "What is LangChain?"
                    ↓
2. Embed Query: Convert to 1536-dimensional vector using embeddings model
                    ↓
3. Vector Search: Find similar vectors in vector database
                    ↓
4. Retrieve Documents: Get top-5 most similar documents/chunks
                    ↓
5. Format Context: Create prompt like:
   "Context: [Retrieved docs...]
    Question: What is LangChain?
    Answer:"
                    ↓
6. LLM Generation: Send to GPT with context
                    ↓
7. Return Answer: "LangChain is a framework for building..."
```

### RAG vs. Traditional LLMs

| Aspect | Traditional LLM | RAG |
|--------|-----------------|-----|
| **Knowledge Source** | Pre-trained weights | Real-time knowledge base |
| **Data Currency** | Fixed (training date) | Always up-to-date |
| **Hallucinations** | Common (makes up facts) | Minimal (grounded in docs) |
| **Proprietary Data** | Can't access | Full access |
| **Cost** | High fine-tuning cost | Cheaper (just retrieve) |
| **Update Speed** | Weeks (retrain) | Instant (add to DB) |
| **Accuracy** | 70-80% on unknown data | 85-95% (depends on retrieval) |
| **Traceability** | "Black box" | Can show source documents |

---

## Why RAG? (Comprehensive Benefits)

### 1. **Addresses LLM Knowledge Limitations**
LLMs have:
- **Knowledge cutoff date** (GPT-3.5 trained on data until April 2023)
- **Limited context window** (4K-128K tokens at most)
- **Can't know your proprietary data** (internal documents, databases, etc.)

RAG solves this by injecting real, current information.

```python
# Without RAG: LLM doesn't know your company's policies
query = "What is our vacation policy?"
response = llm.predict(query)
# Output: "I don't have information about your company's policies..."

# With RAG: Retrieve company handbook first
context = vector_store.similarity_search("vacation policy", k=3)
enhanced_query = f"Context: {context}\n\nQ: {query}"
response = llm.predict(enhanced_query)
# Output: "Your company offers 25 days paid vacation annually..."
```

### 2. **Reduces Hallucinations**
Hallucinations occur when LLMs confidently generate false information.

```python
# Example hallucination
query = "What features does product XYZ have?"

# Without RAG (high hallucination risk)
response = llm.predict(query)
# Output: "Product XYZ has AI integration, blockchain support, 
#          quantum computing..." (Some/all false!)

# With RAG (grounded in real product docs)
docs = vector_store.similarity_search("XYZ features", k=5)
response = llm.predict(query, context=docs)
# Output: Only mentions features in actual documentation
```

### 3. **Lower Cost Than Fine-Tuning**

| Approach | Cost | Speed | Flexibility |
|----------|------|-------|-------------|
| **Fine-tuning** | $1000s | Weeks | Low (need to retrain) |
| **RAG** | $10s-100s | Hours | High (just add docs) |
| **Both** | Medium | Days | Highest |

Fine-tuning is expensive and slow. RAG lets you update knowledge instantly.

### 4. **Keeps Knowledge Fresh**
- **Fine-tuning**: Train once, data becomes stale
- **RAG**: Update database = instant knowledge refresh

```python
# Company updates handbook
old_handbook = "vacation: 20 days"
new_handbook = "vacation: 25 days"

# With fine-tuning: Need to retrain model (expensive, slow)
# With RAG: Just update vector store
vector_store.delete(old_document_id)
vector_store.add_documents(new_handbook)
# Immediately returns new answer!
```

### 5. **Improves Reliability & Trust**
- Users can verify answers by seeing sources
- Auditable: Track which documents influenced responses
- Reduces legal/compliance risk (can prove source)

```python
# Show users where information came from
result = qa_chain({"query": "What's the refund policy?"})
print(result["answer"])
print("Sources:")
for doc in result["source_documents"]:
    print(f"- {doc.metadata['source']} (Page {doc.metadata['page']})")
```

### 6. **Handles Private/Proprietary Data**
- Internal documents never exposed to model weights
- Data stays in database, not in model
- Complies with data privacy (GDPR, CCPA)

### 7. **Enables Real-Time Knowledge Updates**
```python
# Breaking news or urgent updates
new_info = "COVID lockdown in Shanghai lifted"
vector_store.add_documents([new_info])

# Immediately available in answers
response = qa_chain.run("What's the latest on COVID?")
# Returns updated information
```

### 8. **Scalable Knowledge Management**
- Store millions of documents
- Instantly searchable
- No model retraining needed

---

## RAG Pipeline & Components (Detailed)

### Complete RAG Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG System Components                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. DATA PREPARATION LAYER                                       │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────────┐       │
│  │ Document    │→→→│ Chunking     │→→→│ Cleaning &      │       │
│  │ Loaders     │   │ Strategy     │   │ Preprocessing   │       │
│  └─────────────┘   └──────────────┘   └─────────────────┘       │
│          ↓                                       ↓               │
│                                                                   │
│  2. EMBEDDING LAYER                                              │
│  ┌─────────────────────────────────────────────────────┐         │
│  │ Embedding Model (OpenAI/HuggingFace/Cohere)       │         │
│  │ Converts text chunks → 768/1536/3072-dim vectors  │         │
│  └─────────────────────────────────────────────────────┘         │
│          ↓                                                       │
│                                                                   │
│  3. STORAGE LAYER                                                │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                │
│  │ FAISS      │  │ Chroma     │  │ Pinecone   │ ...            │
│  │ (Local)    │  │ (Local/DB) │  │ (Cloud)    │                │
│  └────────────┘  └────────────┘  └────────────┘                │
│          ↓                                                       │
│                                                                   │
│  4. RETRIEVAL LAYER                                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Similarity Search / MMR / Hybrid (Semantic + Keyword)    │  │
│  │ Filter by metadata, apply re-ranking                     │  │
│  └───────────────────────────────────────────────────────────┘  │
│          ↓                                                       │
│                                                                   │
│  5. CONTEXT AUGMENTATION LAYER                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Combine retrieved docs + user query into single prompt   │  │
│  │ Reorder docs by relevance, add metadata hints            │  │
│  └───────────────────────────────────────────────────────────┘  │
│          ↓                                                       │
│                                                                   │
│  6. GENERATION LAYER                                             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ LLM (GPT-4 / Claude / Llama) generates answer            │  │
│  │ based on context + user query                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│          ↓                                                       │
│                                                                   │
│  7. POST-PROCESSING LAYER                                        │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Parse output, format answer, attach sources              │  │
│  │ Validate response quality, collect metrics               │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 7 Core Components Explained

#### 1. **Document Loaders** (Input)
Responsible for ingesting various data sources.

```python
from langchain.document_loaders import (
    PyPDFLoader,      # PDF files
    TextLoader,       # .txt files
    DirectoryLoader,  # Multiple files
    WebBaseLoader,    # Web pages
    CSVLoader,        # CSV data
    JSONLoader        # JSON files
)

# Example
pdf_loader = PyPDFLoader("manual.pdf")
documents = pdf_loader.load()  # Returns list of Document objects
```

**Supported Formats**: PDF, Word, Excel, HTML, JSON, CSV, Web, APIs, Databases

#### 2. **Text Splitters** (Chunking)
Divides documents into manageable chunks while preserving context.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Max characters per chunk
    chunk_overlap=200,    # Overlap between chunks
    separators=["\n\n", "\n", " ", ""]  # Split on these in order
)

chunks = splitter.split_documents(documents)
```

#### 3. **Embeddings Model** (Vectorization)
Converts text to dense numerical vectors.

```python
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Input: "What is AI?"
# Output: [0.123, -0.456, 0.789, ...1536 dimensions...]
vector = embeddings.embed_query("What is AI?")
```

#### 4. **Vector Store** (Storage & Indexing)
Stores embeddings for fast similarity search.

```python
from langchain.vectorstores import Chroma

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Supports: FAISS, Chroma, Pinecone, Weaviate, Milvus, Qdrant...
```

#### 5. **Retriever** (Search)
Finds relevant documents for a given query.

```python
retriever = vector_store.as_retriever(
    search_type="similarity",  # or "mmr"
    search_kwargs={"k": 5}     # Return top-5 results
)

retrieved_docs = retriever.get_relevant_documents("How to reset?")
```

#### 6. **Prompt Template** (Context Formatting)
Formats retrieved context + query into a single prompt.

```python
from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template("""
You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer:
""")

# Formats as: "You are a helpful assistant. Use the following context...
# Context: <retrieved docs>
# Question: <user query>
# Answer:"
```

#### 7. **LLM** (Generation)
Generates final answer based on context and query.

```python
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4",
    temperature=0,  # Deterministic
    max_tokens=500
)

# Takes formatted prompt, returns answer
answer = llm.predict(formatted_prompt)
```

---

## Types of RAG Systems

### 1. **Basic RAG** (Most Common)
Simple retrieve-then-generate approach.

```python
# Flow: Query → Retrieve → Generate
query = "What's in chapter 3?"
docs = vector_store.similarity_search(query, k=3)
prompt = f"Context: {docs}\n\nQ: {query}"
answer = llm.predict(prompt)
```

**Use Case**: Document Q&A, FAQ systems, knowledge bases  
**Pros**: Simple, fast, reliable  
**Cons**: May miss complex multi-document reasoning

---

### 2. **Iterative RAG** (Multi-Turn)
Retrieves multiple times in a loop to refine answers.

```python
# Flow: Query → Retrieve → Generate → Check if good enough
# If not, Retrieve again with better query

query = "How do I troubleshoot error X123?"
iterations = 0

while iterations < 3:
    docs = vector_store.similarity_search(query, k=3)
    answer = llm.predict(f"Context: {docs}\n\nQ: {query}")
    
    # Check if answer is satisfactory
    confidence = check_answer_quality(answer)
    if confidence > 0.8:
        break
    
    # Reformulate query for better retrieval
    query = llm.predict(f"Better search query for: {query}")
    iterations += 1
```

**Use Case**: Complex troubleshooting, research tasks  
**Pros**: Better accuracy through refinement  
**Cons**: Multiple LLM calls = higher cost

---

### 3. **Hierarchical RAG**
Retrieves at multiple levels (document → section → paragraph).

```python
# Step 1: Find relevant documents
relevant_docs = vector_store.similarity_search(query, k=5)

# Step 2: For each doc, find relevant sections
for doc in relevant_docs:
    sections = section_store.similarity_search(query, k=2)
    
# Step 3: For sections, find specific paragraphs
for section in sections:
    paragraphs = paragraph_store.similarity_search(query, k=1)

# Use only the most specific paragraphs
answer = llm.predict(f"Context: {paragraphs}\n\nQ: {query}")
```

**Use Case**: Large documents (books, manuals, policies)  
**Pros**: More precise retrieval  
**Cons**: Complex to implement

---

### 4. **Conditional RAG** (Routing)
Routes queries to different retrieval strategies based on content type.

```python
from langchain.agents import Tool, initialize_agent

# Different retrievers for different types
pdf_retriever = vector_store_pdf.as_retriever()
code_retriever = vector_store_code.as_retriever()
web_retriever = web_search_tool

tools = [
    Tool(name="PDF_Search", func=pdf_retriever.get_relevant_documents),
    Tool(name="Code_Search", func=code_retriever.get_relevant_documents),
    Tool(name="Web_Search", func=web_retriever)
]

# Agent decides which tool to use
agent = initialize_agent(tools, llm, agent_type="openai_functions")
answer = agent.run("Find documentation on feature X")
```

**Use Case**: Multi-source systems (docs + code + web)  
**Pros**: Optimal retrieval for each content type  
**Cons**: Complex routing logic

---

### 5. **Conversational RAG** (Multi-Turn Chat)
Maintains conversation history and uses it for context.

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

memory = ConversationBufferMemory(memory_key="chat_history")

conv_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(),
    memory=memory
)

# First turn
response = conv_chain({"question": "What features does product X have?"})

# Second turn (remembers context)
response = conv_chain({"question": "Which is cheapest?"})
# Knows you're comparing features from product X
```

**Use Case**: Chatbots, assistants, interactive Q&A  
**Pros**: Natural conversation flow  
**Cons**: Need to manage conversation history

---

### 6. **Hybrid RAG** (Semantic + Keyword)
Combines semantic search with traditional keyword search.

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# Semantic (vector-based)
semantic = vector_store.as_retriever(k=3)

# Keyword (BM25)
keyword = BM25Retriever.from_documents(documents)

# Combine both
hybrid = EnsembleRetriever(
    retrievers=[semantic, keyword],
    weights=[0.6, 0.4]  # 60% semantic, 40% keyword
)

docs = hybrid.get_relevant_documents(query)
```

**Use Case**: Technical docs, code search, specific terminology  
**Pros**: Better recall (catches both semantic + exact matches)  
**Cons**: More complex

---

### 7. **Self-Correcting RAG** (Feedback Loop)
Evaluates answer quality and corrects itself.

```python
# Generate answer
answer = llm.predict(f"Context: {docs}\n\nQ: {query}")

# Evaluate: Does answer match retrieved docs?
is_grounded = evaluate_grounding(answer, docs)

if not is_grounded:
    # Answer goes beyond documents, reduce confidence
    answer = f"Based on available documents: {answer}"
    answer += "\n(Note: This response combines information from multiple sources)"

# Or retry with better docs
if not is_grounded:
    docs = vector_store.max_marginal_relevance_search(query, k=5)
    answer = llm.predict(f"Context: {docs}\n\nQ: {query}")
```

**Use Case**: High-accuracy systems, critical applications  
**Pros**: Self-correcting, more reliable  
**Cons**: Requires evaluation logic

---

## Additional RAG Topics

### Retrieval Augmentation Techniques

#### Re-Ranking
Reorder retrieved documents by actual relevance.

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Compress retrieved docs to only relevant parts
compressor = LLMChainExtractor.from_llm(llm, get_prompts.PROMPT)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vector_store.as_retriever(k=10)
)

# Returns only top-3 most relevant docs
compressed_docs = compression_retriever.get_relevant_documents(query)
```

#### Query Expansion
Generate multiple query variations to improve recall.

```python
# Original query
query = "What are deployment strategies?"

# Generate variations
queries = [
    "What are deployment strategies?",
    "How do I deploy applications?",
    "Deployment best practices",
    "CI/CD strategies",
    "Production rollout methods"
]

# Search with all variations
all_docs = []
for q in queries:
    docs = vector_store.similarity_search(q, k=2)
    all_docs.extend(docs)

# Deduplicate and use
unique_docs = {doc.metadata['id']: doc for doc in all_docs}.values()
```

#### Document Summarization
Add summaries as metadata to improve retrieval.

```python
from langchain.chains import load_summarize_chain

# For each document, add summary
for doc in documents:
    summary = load_summarize_chain(llm, chain_type="map_reduce").run([doc])
    doc.metadata["summary"] = summary

# Better retrieval using summaries
retriever = vector_store.as_retriever(search_kwargs={
    "k": 5,
    "fetch_k": 10,  # Get 10, rerank to 5
    "use_summary": True
})
```

### Evaluation & Quality Metrics

```python
# Measure retrieval quality
def evaluate_rag(question, retrieved_docs, ground_truth_docs):
    # Precision: fraction of retrieved docs that are relevant
    precision = len(set(retrieved_docs) & set(ground_truth_docs)) / len(retrieved_docs)
    
    # Recall: fraction of relevant docs that were retrieved
    recall = len(set(retrieved_docs) & set(ground_truth_docs)) / len(ground_truth_docs)
    
    # F1 score: harmonic mean
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return {"precision": precision, "recall": recall, "f1": f1}

# Measure answer quality
def evaluate_answer(answer, retrieved_docs):
    # Is answer grounded in docs?
    grounding_score = check_if_grounded(answer, retrieved_docs)
    
    # Is answer accurate?
    accuracy_score = verify_against_truth(answer)
    
    # Is answer complete?
    completeness_score = check_coverage(answer, query)
    
    return grounding_score + accuracy_score + completeness_score
```

### Caching & Performance Optimization

```python
from langchain.cache import InMemoryCache, SQLiteCache
import langchain

# Cache embeddings to avoid recomputation
langchain.llm_cache = SQLiteCache(database_path="./llm_cache.db")

# Cache retrieval results
retrieval_cache = {}

def cached_retrieval(query, k=3):
    cache_key = (query, k)
    if cache_key not in retrieval_cache:
        retrieval_cache[cache_key] = vector_store.similarity_search(query, k)
    return retrieval_cache[cache_key]
```

### Monitoring & Logging

```python
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def logged_rag_pipeline(query):
    logger.info(f"[{datetime.now()}] Query received: {query}")
    
    # Retrieve
    docs = vector_store.similarity_search(query, k=3)
    logger.info(f"Retrieved {len(docs)} documents")
    logger.info(f"Retrieval scores: {[doc.metadata.get('score', 'N/A') for doc in docs]}")
    
    # Generate
    answer = llm.predict(f"Context: {docs}\n\nQ: {query}")
    logger.info(f"Generated answer length: {len(answer)} chars")
    
    return answer
```

---

## 1. Embedding Models

Embeddings convert text into numerical vectors that capture semantic meaning.

### OpenAI Embeddings
```python
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
# Cost: $0.02 per 1M tokens
# Dimensions: 3072
# Quality: Excellent, but most expensive
```

### HuggingFace Embeddings
```python
from langchain.embeddings import HuggingFaceEmbeddings

# Free, local, no API calls
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Dimensions: 384
# Speed: Fast, runs locally
# Quality: Good for most tasks
```

### Cohere Embeddings
```python
from langchain.embeddings import CohereEmbeddings

embeddings = CohereEmbeddings(model="embed-english-v3.0")
# Cost: $0.10 per 1M tokens
# Dimensions: 1024
# Quality: Very good, multilingual support
```

### Comparison

| Model | Provider | Cost | Dimensions | Speed | Quality |
|-------|----------|------|-----------|-------|---------|
| text-embedding-3-large | OpenAI | High | 3072 | Moderate | Excellent |
| sentence-transformers | HuggingFace | Free | 384 | Very Fast | Good |
| embed-english-v3.0 | Cohere | Medium | 1024 | Moderate | Very Good |
| instructor-large | HuggingFace | Free | 768 | Fast | Excellent |

**Best Practice**: Start with HuggingFace for prototyping; upgrade to OpenAI/Cohere for production.

---

## 2. Document Chunking Strategies

How you split documents dramatically affects retrieval quality.

### Character Text Splitter (Simple)
```python
from langchain.text_splitter import CharacterTextSplitter

# Works: Split by character count
splitter = CharacterTextSplitter(
    chunk_size=1000,      # Characters per chunk
    chunk_overlap=200     # Overlap between chunks
)

chunks = splitter.split_documents(documents)
```

**Use Case**: Simple documents, web pages.

### Recursive Character Text Splitter (Better)
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Better: Splits on semantic boundaries
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]  # Try these first
)

chunks = splitter.split_documents(documents)
```

**Use Case**: Most documents (books, articles, code).

### Semantic Chunking (Advanced)
```python
from langchain.text_splitter import SemanticSplitter
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()

# Splits based on semantic similarity, not character count
semantic_splitter = SemanticSplitter(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile"
)

chunks = semantic_splitter.split_documents(documents)
```

**Use Case**: Complex documents, technical papers, maintaining context.

### Language-Specific Splitter
```python
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter

# Split Python code intelligently
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=500,
    chunk_overlap=50
)

code_chunks = python_splitter.split_documents(code_documents)
```

**Chunking Best Practices:**
- **chunk_size**: 500-1500 chars typically good
- **chunk_overlap**: 10-20% of chunk_size prevents lost context
- **Semantic splittin**g: Best quality but slower
- **Consider metadata**: Preserve document source/page numbers

---

## 3. Vector Store Types

### FAISS (Facebook AI Similarity Search)
```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(documents, embeddings)

# Save locally
vector_store.save_local("faiss_index")

# Load from disk
loaded_store = FAISS.load_local("faiss_index", embeddings)
```

**Pros**: Fast, free, local, no internet needed  
**Cons**: In-memory only (doesn't scale to millions), no filtering  
**Best For**: Development, small datasets, prototyping

---

### Chroma (Modern, Developer-Friendly)
```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# Persistent on disk
vector_store = Chroma.from_documents(
    documents,
    embeddings,
    persist_directory="chroma_data"
)

# Query with filtering
results = vector_store.similarity_search_with_score(
    "How to reset?",
    k=3,
    where={"source": "manual.pdf"}  # Metadata filtering
)
```

**Pros**: Persistent storage, metadata filtering, great API  
**Cons**: Smaller scale than Pinecone  
**Best For**: Small to medium projects, metadata filtering needed

---

### Pinecone (Cloud-Based, Scalable)
```python
from langchain.vectorstores import Pinecone
import pinecone

# Initialize
pinecone.init(api_key="YOUR_API_KEY", environment="us-west1-gcp")

# Create index
pinecone.create_index(
    name="documents",
    dimension=1536,
    metric="cosine"
)

# Use with LangChain
vector_store = Pinecone.from_documents(
    documents,
    embeddings,
    index_name="documents"
)

# Query
results = vector_store.similarity_search("Reset instructions", k=3)
```

**Pros**: Highly scalable (billions of vectors), fast, cloud-managed  
**Cons**: Requires API key, costs money for production  
**Best For**: Production systems, millions+ of documents, real-time requirements

---

### Weaviate (Open-Source, Enterprise)
```python
from langchain.vectorstores import Weaviate
import weaviate

# Connect to Weaviate instance
client = weaviate.Client("http://localhost:8080")

# Create vector store
vector_store = Weaviate.from_documents(
    documents,
    embeddings,
    client=client,
    index_name="Documents"
)

# GraphQL queries possible
results = vector_store.similarity_search("Topic", k=3)
```

**Pros**: Open-source, self-hosted, semantic search + keyword search  
**Cons**: More complex setup than Chroma  
**Best For**: Enterprise needs, hybrid search requirements

---

### Comparison Table

| Store | Type | Scale | Cost | Metadata | Speed |
|-------|------|-------|------|----------|-------|
| **FAISS** | Local | Small | Free | Basic | Very Fast |
| **Chroma** | Local/Cloud | Small-Medium | Free | Yes | Fast |
| **Pinecone** | Cloud | Large | $$ | Yes | Very Fast |
| **Weaviate** | Self-hosted | Medium-Large | Free | Yes | Fast |
| **Milvus** | Self-hosted | Large | Free | Yes | Very Fast |

---

## 4. Retrieval Methods

### Similarity Search
```python
# Standard: Find most similar documents
results = vector_store.similarity_search(
    query="How do I restart?",
    k=3  # Return top 3
)

for doc in results:
    print(doc.page_content)
```

### Similarity Search with Score
```python
# Get similarity scores to evaluate relevance
results = vector_store.similarity_search_with_score(
    query="Reset the device",
    k=3
)

for doc, score in results:
    print(f"Score: {score}, Content: {doc.page_content}")
# Score closer to 1.0 = more similar
```

### Maximum Marginal Relevance (MMR)
```python
# Returns diverse results (not just most similar)
# Useful to avoid redundant/duplicate results
results = vector_store.max_marginal_relevance_search(
    query="Troubleshoot errors",
    k=3,
    fetch_k=10  # Fetch 10, then select 3 diverse ones
)

# MMR balances: relevance + diversity
```

**When to use:**
- **Similarity**: Standard Q&A
- **Similarity with Score**: Decide whether to use retrieved docs
- **MMR**: Avoid repetition, want diverse perspectives

### Metadata Filtering
```python
# Filter by document properties
results = vector_store.similarity_search(
    query="Reset",
    k=3,
    where={"source": "admin_guide.pdf"}  # Chroma syntax
)

# Or with expression filters
results = vector_store.similarity_search(
    query="Reset",
    k=3,
    where={
        "source": "admin_guide.pdf",
        "page": {"$gte": 5}  # Page >= 5
    }
)
```

**Use Case**: Multi-tenant apps, document-specific search.

---

## 5. RetrievalQA Chain Types

Different strategies for injecting retrieved documents into the LLM.

### "Stuff" (Simple Concatenation)
```python
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    chain_type="stuff",  # Default
    retriever=vector_store.as_retriever(k=3)
)

answer = qa_chain.run("How to reset?")
```

**How it works:** Concatenates all retrieved docs into one prompt  
**Pros**: Simple, fast, preserves context  
**Cons**: Fails if total tokens exceed limit  
**Best For**: Small documents or low retrieval counts

---

### "Map-Reduce"
```python
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    chain_type="map_reduce",
    retriever=vector_store.as_retriever(k=10)
)

answer = qa_chain.run("Summarize all troubleshooting steps")
```

**How it works:**
1. Map: Process each document independently
2. Reduce: Combine summaries into final answer

**Pros**: Handles many/large documents, parallel processing  
**Cons**: Loses some context between documents, more expensive  
**Best For**: Long documents, summarization tasks

---

### "Refine"
```python
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    chain_type="refine",
    retriever=vector_store.as_retriever(k=5)
)

answer = qa_chain.run("What are the features?")
```

**How it works:**
1. Process first document
2. Refine answer with each additional document
3. Iterative improvement

**Pros**: Maintains context, better quality for complex queries  
**Cons**: Slower (sequential), more expensive  
**Best For**: Complex questions, high-quality answers needed

---

### Chain Type Comparison

| Type | Speed | Quality | Cost | Context | Use Case |
|------|-------|---------|------|---------|----------|
| **Stuff** | Very Fast | Good | Low | Preserved | Simple Q&A |
| **Map-Reduce** | Fast | Fair | Medium | Lost | Summarization |
| **Refine** | Slow | Excellent | High | Preserved | Complex queries |

---

## 6. Hybrid Search (Keyword + Semantic)

Combines traditional keyword search with semantic similarity for better results.

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import FAISS

# Traditional keyword search (BM25)
bm25_retriever = BM25Retriever.from_documents(documents)

# Semantic search
vector_store = FAISS.from_documents(documents, embeddings)
semantic_retriever = vector_store.as_retriever(k=3)

# Combine both (weights: 0.5 keyword, 0.5 semantic)
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, semantic_retriever],
    weights=[0.5, 0.5]
)

# Use in QA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=ensemble_retriever
)

answer = qa_chain.run("Find error handling code")
```

**When to use:**
- Specific keywords mixed with semantic meaning
- User might search by exact terms
- Technical documentation (code + concepts)

---

## 7. Document Preprocessing

Enhance documents before embeddings.

```python
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader

# Add metadata to preserve context
documents = []
for page_num, doc in enumerate(docs):
    doc.metadata["page"] = page_num
    doc.metadata["source"] = "user_guide.pdf"
    documents.append(doc)

# Clean text (remove extra whitespace)
def clean_text(text):
    import re
    text = re.sub(r'\s+', ' ', text)  # Multi-space → single space
    return text.strip()

for doc in documents:
    doc.page_content = clean_text(doc.page_content)

# Add summaries as metadata (helps with context)
from langchain.chains.summarize import load_summarize_chain

summaries = load_summarize_chain(llm, chain_type="map_reduce")
for doc in documents:
    summary = summaries.run([Document(page_content=doc.page_content)])
    doc.metadata["summary"] = summary
```

---

## 8. Vector Store Updates & Maintenance

```python
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader

# Initial creation
vector_store = FAISS.from_documents(documents, embeddings)
vector_store.save_local("faiss_data")

# Later: Add new documents
new_docs = loader.load_and_split()
vector_store.add_documents(new_docs)
vector_store.save_local("faiss_data")  # Persist changes

# Delete documents (Chroma example)
from langchain.vectorstores import Chroma

chroma_store = Chroma(embed_function=embeddings, persist_directory="./chroma")
chroma_store.delete(ids=["doc_123"])  # Delete by ID
chroma_store.persist()
```

---

## 9. Complete RAG Pipeline

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Step 1: Load documents
pdf_loader = PyPDFLoader("documentation.pdf")
documents = pdf_loader.load()

# Step 2: Chunk documents smartly
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(documents)

# Step 3: Add metadata
for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = i

# Step 4: Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 5: Store in vector database
vector_store = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="./chroma_docs"
)

# Step 6: Create retriever with filtering
retriever = vector_store.as_retriever(
    search_type="mmr",  # Maximum Marginal Relevance
    search_kwargs={"k": 5, "fetch_k": 10}
)

# Step 7: Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    chain_type="refine",  # Better for complex docs
    retriever=retriever,
    return_source_documents=True
)

# Step 8: Query
result = qa_chain({"query": "What's the API rate limit?"})
print("Answer:", result["result"])
print("Sources:", result["source_documents"])
```

---

## 10. RAG Best Practices

| Practice | Why | Example |
|----------|-----|---------|
| **Use semantic chunking** | Better context preservation | Split by paragraphs, not arbitrary chars |
| **Add metadata** | Enables filtering & traceability | source, date, author, page_number |
| **Test chunk sizes** | Affects quality & cost | Try 500, 1000, 1500 char chunks |
| **Use MMR for diversity** | Avoid redundant results | `max_marginal_relevance_search()` |
| **Store & version vectors** | Reproducibility & rollback | Save embeddings with timestamps |
| **Monitor retrieval quality** | Catch issues early | Log similarity scores, track failures |
| **Hybrid search** | Combines strengths | Keyword + semantic together |
| **Add document summaries** | Better context in metadata | Use LLM to summarize each doc |
| **Implement caching** | Reduce API costs | Cache embeddings for identical queries |
| **Batch document updates** | Efficiency | Update in bulk, not one-by-one |

---

## 11. Troubleshooting RAG Issues

**Issue**: Low retrieval quality
- **Solution**: Try semantic chunking, adjust chunk_size, use better embeddings

**Issue**: Token limit exceeded
- **Solution**: Use map_reduce chain, reduce k (number of retrieved docs), shorter chunks

**Issue**: Slow retrievals
- **Solution**: Use FAISS/Pinecone, reduce fetch_k, use approximate search

**Issue**: Irrelevant results
- **Solution**: Use MMR, add better metadata, filter by source

**Issue**: High embedding costs
- **Solution**: Use HuggingFace embeddings locally, batch embedding calls

---

## 12. RAG with Sources

```python
from langchain.chains import RetrievalQA

# Get sources in response
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True  # Enable sources
)

result = qa_chain({"query": "What's the warranty?"})

print("Answer:", result["result"])
print("\nSources:")
for doc in result["source_documents"]:
    print(f"- {doc.metadata['source']}, Page {doc.metadata['page']}")
```

**Use Case**: Show users where information came from (transparency, verification).

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

