

# 🚍 AI-Powered Public Transport Chatbot & Booking System (with RAG)

## 1. Project Overview
This project builds an **AI-powered conversational platform** that allows passengers to search, inquire, and book public transport services using natural language.  

With **Retrieval-Augmented Generation (RAG)**, the chatbot retrieves transport data (routes, schedules, seat availability) from a vector database (Chroma) and generates natural language responses using LLMs (OpenAI GPT-4).

**Key Features:**
- 🔍 Natural language bus search with RAG
- 🚌 Real-time seat availability tracking
- 🎟️ Multi-agent booking workflow
- 💳 Payment processing (dummy)
- 🎫 Automatic ticket generation

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     USER INTERFACE                       │
│              (Chat Input / WhatsApp / Web)               │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │   TransportBookingAssistant        │
        │   (Main Orchestrator)              │
        └────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
    ┌────────┐   ┌─────────────┐   ┌──────────┐
    │  RAG   │   │   LangGraph │   │  Agents  │
    │ Chroma │   │   Pipeline  │   │  Layer   │
    │ Vector │   │             │   │          │
    │  Store │   └─────────────┘   └──────────┘
    └────────┘
        │
        ▼
┌─────────────────────────────────────┐
│    Multi-Agent Booking Pipeline     │
├─────────────────────────────────────┤
│ 1. SearchBusAgent (RAG search)      │
│ 2. SelectionAgent (Bus selection)   │
│ 3. ConfirmBusAgent (Confirmation)   │
│ 4. BookingAgent (Booking details)   │
│ 5. PaymentAgent (Payment process)   │
│ 6. ConfirmationAgent (Ticket gen)   │
└─────────────────────────────────────┘
```

---

## 3. Use Case Flow (End-to-End Booking Journey)

### Step 1: Bus Search (SearchBusAgent)
```
User: "I need a bus from Tambaram to Kilambakkam"
     ↓
     ├─ Query entered into RAG system
     ├─ Chroma vector DB retrieves matching buses
     ├─ BusDataParser extracts route, time, price, seats
     └─ System displays: Bus 451 | Route: Tambaram → Kilambakkam | Departure: 2:15 PM | Price: ₹120 | Available: 12 seats
```

**Process:**
- Query is embedded and searched in Chroma vector database
- Retriever returns all matching bus documents
- Locations extracted from natural language query
- Bus information parsed and formatted
- User sees numbered list of available buses

### Step 2: Bus Selection (SelectionAgent)
```
System: "👉 Please select a bus (enter the number 1-3)"
User: "1"
     ↓
     ├─ Selection validated (numeric, in range)
     ├─ Selected bus details extracted
     ├─ Bus number and price stored in state
     └─ ✅ Selected Bus 451
```

**Process:**
- User inputs selection number
- SelectionAgent validates numeric input
- Bus details and price extracted
- State updated with selected bus

### Step 3: Bus Confirmation (ConfirmBusAgent)
```
System: "Please confirm bus selection - Bus 451 to Kilambakkam?"
User: "Yes"
     ↓
     ├─ Confirmation processed
     ├─ YES → Proceed to booking
     └─ NO → Return to bus search
```

**Process:**
- Confirms user's bus selection
- Routes yes/no responses appropriately
- Prevents accidental bookings

### Step 4: Booking Details (BookingAgent)
```
System: "How many tickets do you need?"
User: "2"
     ↓
     ├─ Collect passenger count
     ├─ Calculate total fare (₹120 × 2 = ₹240)
     └─ Confirm booking details
```

**Process:**
- Collects passenger/ticket count
- Calculates total amount
- Prepares booking for payment

### Step 5: Payment Processing (PaymentAgent)
```
System: "Processing payment of ₹240..."
     ↓
     ├─ Payment validated (dummy)
     ├─ Transaction ID generated: TXN_2024_001
     ├─ Payment confirmed: ✅
     └─ Proceed to ticket generation
```

**Process:**
- Validates payment amount
- Generates transaction ID
- Confirms payment status
- Prevents double-booking

### Step 6: Ticket Generation (ConfirmationAgent)
```
System: "🎫 BOOKING CONFIRMED
         Booking Reference: BK_451_12345
         Bus: 451 | Route: Tambaram → Kilambakkam
         Date: 2024-01-15 | Departure: 2:15 PM
         Passengers: 2 | Total: ₹240
         Payment: Confirmed (TXN_2024_001)"
```

**Process:**
- Generates booking reference
- Creates final ticket
- Shows complete booking summary
- Booking complete ✅

---

## 4. Available Dataset

| Field             | Description                              |
| ─────────────────────────────────────────────────────────────── |
| Bus Number        | Unique bus identifier (e.g., 451, 452)  |
| Route             | Start and end locations                  |
| Intermediate Stops| All stops between origin and destination |
| Departure Time    | Scheduled departure time (12-hr format)  |
| Arrival Time      | Expected arrival time                    |
| Fare              | Price per ticket (₹)                     |
| Bus Type          | Volvo, Sleeper, Electric, City Bus, etc. |
| AC/Non-AC         | Bus category (AC or Non-AC)              |
| Seat Availability | Available seats for booking              |

---

## 5. Setup Instructions

### Prerequisites
- Python 3.8+
- OpenAI API Key
- Chroma Vector Database
- LangChain & LangGraph

### Installation Steps

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

**Key packages:**
```
langchain==0.1.0+
langchain-openai==0.1.0+
langchain-chroma==0.1.0+
langgraph==0.0.1+
fastapi==0.100+
python-dotenv==1.0+
```

#### 2. Setup Environment Variables
Create `.env` file:
```bash
OPENAI_API_KEY=your_api_key_here
```

#### 3. Prepare Dataset
The system uses `chennai_transport_dataset.json` with bus route data:
```bash
python dataset_prepartion.py  # Prepare data
python transportBusIndexer.py # Index into Chroma DB
```

#### 4. Run the Application

**Option A: Direct Python Script**
```bash
python searchTransport.py
```

**Option B: FastAPI Server**
```bash
uvicorn api:app --reload
```

---

## 6. Project File Structure

| File | Purpose | Status |
|───────────────────────────────────────────────────────────|
| **searchTransport.py** | Main orchestrator & multi-agent coordinator | ✅ Active |
| **agents.py** | 6 specialized agent classes (Search, Select, Confirm, Book, Pay, Confirm) | ✅ Active |
| **prompts.py** | System prompts, templates, and instruction sets | ✅ Active |
| **api.py** | FastAPI endpoints for web integration | ✅ Active |
| **bus_parser.py** | Bus data extraction & formatting utilities | ✅ Active |
| **chat_bot.py** | Conversational interface | ✅ Ready |
| **transportBusIndexer.py** | Vector DB indexing pipeline | ✅ Active |
| **dataset_prepartion.py** | Data preprocessing & transformation | ✅ Active |
| **test_booking_flow.py** | Integration tests for booking pipeline | ✅ Testing |
| **test_validation.py** | Unit tests for data validation | ✅ Testing |
| **chennai_transport_dataset.json** | Bus routes & schedule data | ✅ Sample Data |
| **bus_data/** | Chroma vector database storage | ✅ Persisted |

---

## 7. How It Works: Technical Flow

### 7.1 Data Indexing Process
```
Raw JSON Dataset
    ↓
BusDataParser (Extract & enhance)
    ↓
Text Chunks (Embeddings-ready format)
    ↓
OpenAI Embeddings (text-embedding-ada-002)
    ↓
Chroma Vector Store (Persisted in bus_data/)
```

### 7.2 Query Processing & RAG

```
User Query: "Tambaram to Kilambakkam"
    ↓
Query Embedding (OpenAI embeddings)
    ↓
Chroma Semantic Search (Retrieve similar chunks)
    ↓
Retrieved Documents (Top-K matches from vector DB)
    ↓
BusDataParser.extract_bus_info() (Parse structured data)
    ↓
GPT-4 Formatting (Apply system prompts)
    ↓
Formatted Response (Bus 451 | Route: ... | Price: ₹120 | Status: ...)
```

### 7.3 Multi-Agent Orchestration

```
State = {
    query: str,
    user_answer: str,
    bus_list: list,
    bus_number: str,
    amount: float,
    status: str,
    ...
}
    ↓
LangGraph StateGraph Execution
    ├─ search_bus() → available_buses
    ├─ select_bus() → bus_number + amount
    ├─ confirm_bus() → YES/NO routing
    ├─ booking() → passenger details
    ├─ payment() → TXN_ID generation
    └─ confirmation() → Ticket output
    ↓
Final Booking Result
```

### 7.4 Vector DB Schema
**Chroma Collection: "bus_data"**
```
Document:
  content: "Bus 451 AC Volvo departs Tambaram at 2:15 PM..."
  metadata:
    - bus_number: "451"
    - route: "Tambaram → Kilambakkam"
    - departure_time: "2:15 PM"
    - price: "₹120"
    - available_seats: "12"
    - bus_type: "AC Volvo"
```

---

## 8. Usage Examples

### Example 1: Simple Bus Search
```python
from searchTransport import TransportBookingAssistant

assistant = TransportBookingAssistant(persist_dir="bus_data")
result = assistant.run("Tambaram to Kilambakkam", "")
print(result.get("result"))
```

### Example 2: Complete Booking Flow
```python
assistant = TransportBookingAssistant()

# Step 1: Search
result1 = assistant.run("Chennai to Bangalore", "")
print(result1["result"])

# Step 2: Select bus #1
result2 = assistant.run("", "1")
print(result2["result"])

# Step 3: Confirm
result3 = assistant.run("", "yes")
print(result3["result"])
```

### Example 3: FastAPI Integration
```bash
curl -X POST "http://localhost:8000/run" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Tambaram to Kilambakkam",
    "session_id": "user_123"
  }'
```

---

## 9. Key Technologies

| Component | Technology | Purpose |
|────────────────────────────────────────────────────────|
| **LLM** | OpenAI GPT-4 | Query understanding & response generation |
| **Vector DB** | Chroma | Semantic search & RAG retrieval |
| **Embeddings** | OpenAI text-embedding-ada-002 | Vector encoding |
| **Orchestration** | LangGraph | Multi-agent workflow management |
| **Framework** | LangChain | LLM integration & utilities |
| **API** | FastAPI | Web service endpoints |
| **Data** | JSON | Bus route dataset |

---

## 10. Agent Specifications

### SearchBusAgent
- **Input:** User query (natural language)
- **Process:** RAG search + data parsing
- **Output:** List of matching buses with details
- **Status:** ✅ Active

### SelectionAgent
- **Input:** Bus selection number
- **Process:** Validation + bus extraction
- **Output:** Selected bus details
- **Status:** ✅ Active

### ConfirmBusAgent
- **Input:** Confirmation (yes/no)
- **Process:** Route yes/no to next step
- **Output:** Confirmed bus or return to search
- **Status:** ✅ Active

### BookingAgent
- **Input:** Booking confirmation
- **Process:** Collect passenger details, calculate amount
- **Output:** Booking summary
- **Status:** ✅ Active

### PaymentAgent
- **Input:** Amount and payment approval
- **Process:** Generate TXN_ID, validate payment
- **Output:** Payment confirmation
- **Status:** ✅ Active

### ConfirmationAgent
- **Input:** Successful payment
- **Process:** Generate booking reference & ticket
- **Output:** Final ticket and summary
- **Status:** ✅ Active

---

## 11. Troubleshooting

| Issue | Solution |
|──────────────────────────────────────────────────────────|
| "OPENAI_API_KEY not found" | Create .env file with API key |
| "Chroma collection not found" | Run `python transportBusIndexer.py` to index data |
| "No buses found" | Verify query location names match dataset |
| "GPT-4 not available" | Check OpenAI API access/quota |
| "Vector DB connection error" | Ensure `bus_data/` directory exists |

---

## 12. References

- See [MULTI_AGENT_README.md](MULTI_AGENT_README.md) for detailed agent architecture
- See [stpes.md](stpes.md) for next steps and data preparation guide

---

## 13. System Architecture (with RAG)

Passenger  
↓  
Web / Mobile / WhatsApp Chat  
↓  
**AI Agent Layer (LLM + RAG)**  
- Query Understanding (LLM)  
- Retriever (Vector DB: bus schedules, routes, seats)  
- Generator (LLM response)  
↓  
Business APIs  
├── Search Bus API  
├── Route API  
├── Seat Availability API  
├── Booking API  
├── Cancellation API  
└── Payment API  
↓  
Transport Database  

---

## 6. Functional Modules

### Module 1: AI Chatbot (RAG-enabled)
- Understands passenger queries in natural language.
- Uses **RAG pipeline**:
  - Extract intent → Retrieve bus data → Generate response.
- Example:  
  *“Find AC buses from Kelambakam to Kilambakam after 6 PM.”*  
  → Retrieves bus records → Confirms seat availability → Responds naturally.

### Module 2: Bus Search Engine
- Filters: Source, Destination, Date, Time, AC/Non-AC, Bus Type, Seat Availability.
- Output: Bus details, seats, route info, timings, fare.

### Module 3: Seat Availability Management
- Real-time inventory tracking.
- Seat reservation locking.
- Overbooking prevention.

### Module 4: Booking Engine
- Workflow: Search → Verify seats → Confirm booking → Payment → Ticket generation.

---

## 7. Database Design
- **Bus Table** → Bus metadata.  
- **Route Table** → Source/Destination.  
- **Stops Table** → Intermediate stops.  
- **Inventory Table** → Seat availability per journey date.  
- **Booking Table** → Passenger bookings.  

---

## 8. AI Agent Workflow (RAG Example)

User:  
*"I need 2 AC seats from Whitefield to Mysore tomorrow."*

AI Actions:  
1. Extract travel requirements (LLM).  
2. Retrieve bus records (Vector DB).  
3. Call **Search Bus API**.  
4. Display available buses.  
5. Verify seat availability.  
6. Create booking.  
7. Initiate payment.  
8. Generate ticket.  
9. Send confirmation.  

---

## 10. MVP Scope
✅ AI Chatbot (RAG-enabled)  
✅ Bus Search  
✅ Seat Availability  
✅ Booking Management  
✅ Payment Integration  
✅ Ticket Generation  

---

## 11. Expected Outcome
Passengers can search, inquire, and book public transport services entirely through a conversational interface powered by **RAG**.  
The system retrieves accurate transport data, generates natural responses, improves booking efficiency, and scales to multilingual support.

---

