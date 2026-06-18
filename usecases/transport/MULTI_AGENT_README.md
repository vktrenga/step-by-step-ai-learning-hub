# 🚍 Multi-Agent Transport Booking System

## Architecture Overview

This system uses a **specialized multi-agent architecture** where each step of the booking process is handled by a dedicated agent with specific responsibilities.

### Agent Flow

```
┌─────────────┐
│ User Input  │
└──────┬──────┘
       │
       ▼
┌─────────────────────────┐
│  SearchBusAgent         │  ✅ NEW AGENT
│  • Find buses (RAG)     │
│  • Use SEARCH_TEMPLATE  │
│  • Log operations       │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│  ConfirmBusAgent        │  ✅ NEW AGENT  
│  • Ask confirmation    │
│  • Use CONFIRM_TEMPLATE │
│  • Route yes/no        │
└──────┬──────────────────┘
       │
    YES│ NO
       ├──────────────────────────┐
       ▼                          ▼
   ┌─────────────────────┐   [END]
   │  BookingAgent       │   ❌
   │  • Confirm booking  │
   │  • Collect details  │
   │  • Use BOOKING_TEMP │
   └──────┬──────────────┘
          │
       YES│ NO
          ├──────────────────────────┐
          ▼                          ▼
      ┌─────────────────────┐   [END]
      │  PaymentAgent       │   ❌
      │  • Handle payment   │
      │  • Use PAYMENT_TEMP │
      │  • Create TXN ID    │
      └──────┬──────────────┘
             │
          YES│ NO
             ├──────────────────────────┐
             ▼                          ▼
         ┌──────────────────────┐   [END]
         │ ConfirmationAgent    │   ❌
         │ • Generate ticket    │
         │ • Booking ref        │
         │ • Show summary       │
         └──────────────────────┘
                  │
                  ▼
              [END] ✅
```

---

## 📋 Files Overview

### Core Files

| File | Purpose | Status |
|------|---------|--------|
| **searchTransport.py** | Main orchestrator | ✅ Updated |
| **agents.py** | Specialized agent classes | ✅ NEW |
| **prompts.py** | System prompts & templates | ✅ Enhanced |
| **example_agents.py** | Usage examples | ✅ NEW |
| **api.py** | FastAPI endpoints | ✅ Ready |

---

## 🤖 Agent Classes

### 1. **SearchBusAgent** (in searchTransport.py)
```python
def search_bus(self, state: dict):
    # Finds buses using Chroma RAG
    # Uses SEARCH_TEMPLATE with system prompt
    # Logs all operations
```

**Responsibilities:**
- Query Chroma vector DB
- Generate natural response using LLM
- Return bus_number and bus_details

**Output:**
```json
{
    "result": "Found buses...",
    "bus_number": "451",
    "bus_details": "Bus 451 - Available"
}
```

---

### 2. **ConfirmBusAgent** (in agents.py)
```python
class ConfirmBusAgent:
    def process(self, state: dict) -> dict:
        # Asks passenger to confirm selected bus
        # Uses CONFIRM_TEMPLATE
```

**Responsibilities:**
- Confirm bus selection
- Ask passenger for yes/no
- Provide bus summary

**Methods:**
- `process(state)` - Generate confirmation message
- `should_proceed(user_answer)` - Check if user said "yes"

**Output:**
```json
{
    "result": "Confirmation message...",
    "bus_number": "451",
    "status": "awaiting_confirmation",
    "agent": "ConfirmBusAgent"
}
```

---

### 3. **BookingAgent** (in agents.py)
```python
class BookingAgent:
    def process(self, state: dict) -> dict:
        # Process booking and collect details
        # Uses BOOKING_TEMPLATE
```

**Responsibilities:**
- Confirm booking intent
- Collect passenger details (name, contact)
- Generate booking reference
- Ask for final confirmation

**Methods:**
- `process(state)` - Generate booking message
- `should_proceed_to_payment(user_answer)` - Check if ready for payment

**Output:**
```json
{
    "result": "Booking confirmation...",
    "bus_number": "451",
    "booking_ref": "BK451xxxx",
    "status": "pending_payment",
    "agent": "BookingAgent"
}
```

---

### 4. **PaymentAgent** (in agents.py)
```python
class PaymentAgent:
    def process(self, state: dict) -> dict:
        # Handle payment processing
        # Uses PAYMENT_TEMPLATE
```

**Responsibilities:**
- Display payment amount
- List payment methods
- Request payment confirmation
- Generate transaction ID

**Methods:**
- `process(state)` - Generate payment request
- `confirm_payment(payment_method)` - Confirm payment received

**Output:**
```json
{
    "result": "Payment request...",
    "amount": 120,
    "transaction_id": "TXN451xxxxx",
    "status": "payment_pending",
    "agent": "PaymentAgent"
}
```

---

### 5. **ConfirmationAgent** (in agents.py)
```python
class ConfirmationAgent:
    def process(self, state: dict) -> dict:
        # Generate final confirmation
```

**Responsibilities:**
- Generate booking confirmation
- Provide booking reference
- Show booking summary
- Provide support contact

**Output:**
```json
{
    "result": "🎉 Thank you for booking...",
    "booking_ref": "BK451xxxx",
    "status": "booking_confirmed",
    "agent": "ConfirmationAgent"
}
```

---

## 🔄 Conversation Flow Examples

### Example 1: Complete Booking

```
User: "Bus to Bangalore"
→ SearchBusAgent finds buses

Agent: "Bus 451 found. Confirm?"
User: "yes"
→ ConfirmBusAgent processes

Agent: "Proceed with booking?"
User: "yes"
→ BookingAgent processes

Agent: "Pay ₹120?"
User: "yes"
→ PaymentAgent processes

Agent: "🎉 Booking confirmed!"
→ ConfirmationAgent generates ticket
```

### Example 2: User Cancels at Confirmation

```
User: "Bus to Bangalore"
→ SearchBusAgent finds buses

Agent: "Bus 451 found. Confirm?"
User: "no"
→ END (User can search again)
```

### Example 3: User Cancels at Booking

```
User: "Bus to Bangalore"
→ SearchBusAgent finds buses

Agent: "Bus 451 found. Confirm?"
User: "yes"
→ ConfirmBusAgent processes

Agent: "Proceed with booking?"
User: "no"
→ END (User can search again)
```

---

## 🚀 Integration with api.py

The existing `api.py` works perfectly with the new agent system:

```python
from searchTransport import TransportBookingAssistant

sessions = {}

@app.post("/run")
def run_query(request: QueryRequest):
    if request.session_id not in sessions:
        # Creates assistant with all agents
        sessions[request.session_id] = TransportBookingAssistant()
    
    assistant = sessions[request.session_id]
    
    # Each call delegates to appropriate agent
    result = assistant.run(request.query or "", request.user_answer or "")
    return result
```

---

## 📊 Agent Communication

Each agent receives the complete state and updates it:

```python
# Initial state
state = {
    "query": "Bus to Bangalore",
    "user_answer": ""
}

# After SearchBusAgent
state = {
    "query": "Bus to Bangalore",
    "user_answer": "",
    "result": "...",
    "bus_number": "451",
    "bus_details": "..."
}

# After ConfirmBusAgent
state = {
    "query": "Bus to Bangalore",
    "user_answer": "yes",
    "result": "...",
    "bus_number": "451",
    "bus_details": "...",
    "agent": "ConfirmBusAgent",
    "status": "awaiting_confirmation"
}

# ... and so on
```

---

## 🛠️ System Prompts

Each agent uses a specialized system prompt (from `prompts.py`):

| Agent | Prompt | Role |
|-------|--------|------|
| SearchBus | `SEARCH_SYSTEM_PROMPT` | Expert bus assistant |
| ConfirmBus | `CONFIRM_SYSTEM_PROMPT` | Bus Confirmation Agent |
| Booking | `BOOKING_SYSTEM_PROMPT` | Booking Agent |
| Payment | `PAYMENT_SYSTEM_PROMPT` | Payment Processing Agent |

---

## 📝 Logging

All agents log their operations for debugging:

```
INFO:__main__:TransportBookingAssistant initialized with specialized agents
INFO:__main__:run: New conversation started with query: Bus to Bangalore
INFO:__main__:_execute_node: Executing search_bus
INFO:__main__:search_bus: Found 3 buses: ['451', '452', '453']
INFO:__main__:_execute_node: Executing confirm_bus
INFO:__main__:run: Processing user answer: yes
INFO:__main__:_confirm_choice: User confirmed - proceeding to booking
...
```

---

## 🎯 Advantages of Multi-Agent Architecture

| Feature | Benefit |
|---------|---------|
| **Separation of Concerns** | Each agent has single responsibility |
| **Reusability** | Agents can be used independently |
| **Testability** | Easy to unit test individual agents |
| **Maintainability** | Changes to one agent don't affect others |
| **Scalability** | Easy to add new agents |
| **Logging** | Each agent logs its operations |
| **Error Handling** | Agents can handle errors independently |

---

## 🔧 Running Examples

```bash
# Run complete booking example
python example_agents.py

# Or modify example_agents.py to run different scenarios:
# - example_user_rejects()
# - example_multiple_sessions()
# - example_agent_flow()
# - example_with_logging()
```

---


## 🏗️ Architecture Summary

```
TransportBookingAssistant (Orchestrator)
├── SearchBusAgent
│   └── Chroma RAG + SEARCH_TEMPLATE
├── ConfirmBusAgent
│   └── CONFIRM_TEMPLATE + Routing
├── BookingAgent
│   └── BOOKING_TEMPLATE + Details Collection
├── PaymentAgent
│   └── PAYMENT_TEMPLATE + TXN Generation
└── ConfirmationAgent
    └── Final Confirmation + Ticket Generation
```

---


