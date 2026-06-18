import streamlit as st
from searchTransport import TransportBookingAssistant
import re

# Initialize assistant and chat history
if "assistant" not in st.session_state:
    st.session_state.assistant = TransportBookingAssistant()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "passenger_details" not in st.session_state:
    st.session_state.passenger_details = {}

st.title("🚍 Transport Booking Assistant")

# Chat input
user_input = st.chat_input("Type your query or answer...")

def parse_passenger_details(text: str):
    details = {}

    # Try labeled extraction first
    name_match = re.search(r"Name\s*:\s*([A-Za-z ]+)", text, re.I)
    contact_match = re.search(r"Contact\s*Number\s*:\s*(\d+)", text, re.I)
    email_match = re.search(r"Email\s*address\s*:\s*([\w\.-]+@[\w\.-]+)", text, re.I)
    seat_match = re.search(r"Preferred\s*seating\s*:\s*(\w+)", text, re.I)
    
    
    bus_match = re.search(r'\b(?:bus\s*)?(\d+)\b', text, re.IGNORECASE)
    if bus_match:
        choice = int(bus_match.group(1))
        details["bus_number"] = choice
        return details
    if name_match:
        details["name"] = name_match.group(1).strip()
    if contact_match:
        details["contact"] = contact_match.group(1).strip()
    if email_match:
        details["email"] = email_match.group(1).strip()
    if seat_match:
        details["seat"] = seat_match.group(1).strip()

    # Fallback: detect unlabeled values
    if "email" not in details:
        email_fallback = re.search(r"[\w\.-]+@[\w\.-]+", text)
        if email_fallback:
            details["email"] = email_fallback.group(0)

    if "contact" not in details:
        phone_fallback = re.search(r"\b\d{10}\b", text)
        if phone_fallback:
            details["contact"] = phone_fallback.group(0)

    if "seat" not in details:
        seat_fallback = re.search(r"\b(window|aisle)\b", text, re.I)
        if seat_fallback:
            details["seat"] = seat_fallback.group(1).lower()

    # For name, take first word(s) that are not email/phone/seat
    if "name" not in details:
        tokens = text.split()
        for t in tokens:
            if not re.match(r"\d{10}", t) and not re.match(r"[\w\.-]+@[\w\.-]+", t) and t.lower() not in ["window","aisle","yes","no"]:
                details["name"] = t
                break

  
    return details

if user_input:
    assistant = st.session_state.assistant

    # Check if passenger details are provided
    parsed_details = parse_passenger_details(user_input)
    
    # Prioritize bus number selection
    if parsed_details.get("bus_number"):
        print(f"Bus selected: {parsed_details['bus_number']}")
        result = assistant.run("", user_answer=str(parsed_details["bus_number"]))
    # Handle yes/no/payment responses
    elif user_input.lower() in ["yes", "no", "upi", "card", "net banking", "wallet", "1", "2", "3", "4"]:
        result = assistant.run("", user_answer=user_input)
    # Handle other passenger details (name, email, etc)
    elif parsed_details:
        st.session_state.passenger_details.update(parsed_details)
        # Pass the first non-empty value as user answer (e.g., name)
        answer = parsed_details.get("name") or parsed_details.get("contact") or parsed_details.get("email") or user_input
        result = assistant.run("", user_answer=answer)
    # Default: treat as new query
    else:
        result = assistant.run(user_input, "")

    # Save conversation
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Assistant", result.get("result", "No result available")))

# Display chat history in bubbles
for speaker, text in st.session_state.chat_history:
    if speaker == "You":
        st.chat_message("user").write(text)
    else:
        st.chat_message("assistant").write(text)

# Show stored passenger details
# if st.session_state.passenger_details:
#     st.write("### Current Passenger Details")
#     for k, v in st.session_state.passenger_details.items():
#         st.write(f"- {k.capitalize()}: {v}")
