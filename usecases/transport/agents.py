"""
Separate Agent Classes for Transport Booking System
Each agent handles a specific responsibility in the booking workflow
"""

import logging
from typing import Dict, Optional
from langchain_openai import ChatOpenAI
import warnings
warnings.filterwarnings("ignore", message="Accessing `__path__`")
from prompts import (
    CONFIRM_TEMPLATE, BOOKING_TEMPLATE, PAYMENT_TEMPLATE
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# SELECTION AGENT
# ============================================

class SelectionAgent:
    """
    Agent responsible for handling bus selection from the list.
    Manages selection validation and state transitions.
    """
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.name = "SelectionAgent"
    
    def process(self, state: Dict) -> Dict:
        """
        Process bus selection from the available list
        
        Args:
            state: Current state with available_buses and user_answer
            
        Returns:
            Updated state with selected bus or selection message
        """
        user_answer = state.get("user_answer", "").strip()
        available_buses = state.get("available_buses", [])
        
        logger.info(f"{self.name}: Processing selection")
        logger.info(f"{self.name}: Available buses: {len(available_buses)}")
        
        if not available_buses:
            logger.warning(f"{self.name}: No buses available")
            return {
                "result": "No buses available for selection.",
                "status": "error",
                "agent": self.name
            }
        
        # If user hasn't provided input yet
        if not user_answer:
            logger.info(f"{self.name}: No selection yet, showing options")
            return {
                "result": state.get("bus_list", "Please select a bus from the list"),
                "available_buses": available_buses,
                "status": "awaiting_selection",
                "agent": self.name
            }
        
        # Validate numeric selection
        # if not user_answer.isdigit():
        #     logger.warning(f"{self.name}: Invalid input (not numeric): {user_answer}")
        #     return {
        #         "result": f"Invalid input '{user_answer}'. Please enter a number between 1 and {len(available_buses)}",
        #         "available_buses": available_buses,
        #         "status": "invalid_input",
        #         "agent": self.name
        #     }
        
        # Parse selection
        selection_index = int(user_answer) - 1
        if not (0 <= selection_index < len(available_buses)):
            logger.warning(f"{self.name}: Selection out of range: {user_answer}")
            return {
                "result": f"Invalid selection {user_answer}. Please choose a number between 1 and {len(available_buses)}",
                "available_buses": available_buses,
                "status": "invalid_selection",
                "agent": self.name
            }
        
        # Valid selection - extract bus details
        selected_bus = available_buses[selection_index]
        bus_number = selected_bus["bus_number"]
        bus_details = selected_bus["bus_details"]
        
        logger.info(f"{self.name}: Bus selected - {bus_number}")
        
        return {
            "result": f"✅ Selected Bus {bus_number}",
            "bus_number": bus_number,
            "bus_details": bus_details,
            "parsed_info": selected_bus["parsed_info"],
            "available_buses": available_buses,
            "status": "selection_complete",
            "agent": self.name
        }
    
    def is_valid_selection(self, state: Dict) -> bool:
        """Check if a valid selection has been made"""
        return state.get("status") == "selection_complete"


# ============================================
# CONFIRM BUS AGENT
# ============================================

class ConfirmBusAgent:
    """
    Agent responsible for confirming bus selection.
    Asks passenger to verify selected bus details.
    """
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.name = "ConfirmBusAgent"
    
    def process(self, state: Dict) -> Dict:
        """
        Confirm selected bus with passenger
        
        Args:
            state: Current conversation state with bus_number
            
        Returns:
            Updated state with confirmation message
        """
        bus_number = state.get("bus_number", "Unknown")
        bus_details = state.get("bus_details", f"Bus {bus_number} - Check availability")
        
        logger.info(f"{self.name}: Confirming Bus {bus_number}")
        logger.info(f" Bus Detailas {bus_details}")

        try:
            # Use LLM with system prompt for natural response
            prompt = CONFIRM_TEMPLATE.invoke({
                "bus_number": bus_number,
                "bus_details": bus_details
            })
            response = self.llm.invoke(prompt)
            
            result_message = response.content
        except Exception as e:
            logger.error(f"{self.name} error: {str(e)}")
            result_message = f"Bus {bus_number} found. Would you like to proceed? (yes/no)"
        
        return {
            "result": result_message,
            "bus_number": bus_number,
            "agent": self.name,
            "status": "awaiting_confirmation"
        }
    
    def should_proceed(self, user_answer: str) -> bool:
        """Check if user wants to proceed with booking"""
        return "yes" in user_answer.lower()


# ============================================
# BOOKING AGENT
# ============================================

class BookingAgent:
    """
    Agent responsible for processing booking details.
    Collects passenger information and confirms booking details.
    """
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.name = "BookingAgent"
    
    def process(self, state: Dict) -> Dict:
        """
        Process booking and collect passenger details
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated state with booking confirmation message
        """
        bus_number = state.get("bus_number", "Unknown")
        passenger_name = state.get("passenger_name", "")
        user_answer = state.get("user_answer", "")
        
        logger.info(f"{self.name}: Processing booking for Bus {bus_number}")
        
        # If no passenger name yet, ask for it
        if not passenger_name:
            logger.info(f"{self.name}: Requesting passenger name")
            return {
                "result": f"Please enter your name for the booking:",
                "bus_number": bus_number,
                "agent": self.name,
                "status": "awaiting_passenger_info"
            }
        
        # If we have passenger info, confirm booking
        logger.info(f"{self.name}: Booking confirmed for {passenger_name}")
        
        # Generate booking reference
        booking_ref = f"BK{bus_number}{hash(passenger_name) % 10000:04d}"
        
        return {
            "result": f"✅ Booking confirmed for {passenger_name}!\nBooking Reference: {booking_ref}\n\nProceeding to payment...",
            "bus_number": bus_number,
            "passenger_name": passenger_name,
            "booking_ref": booking_ref,
            "agent": self.name,
            "status": "booking_confirmed"
        }
    
    def should_proceed_to_payment(self, user_answer: str) -> bool:
        """Check if user wants to proceed to payment"""
        return "yes" in user_answer.lower()


# ============================================
# PAYMENT AGENT
# ============================================

class PaymentAgent:
    """
    Agent responsible for payment processing.
    Handles payment method selection and confirms payment.
    """
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.name = "PaymentAgent"
        self.amount = 120  # Default amount in rupees
    
    def process(self, state: Dict) -> Dict:
        """
        Process payment request
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated state with payment confirmation message
        """
        bus_number = state.get("bus_number", "Unknown")
        passenger_name = state.get("passenger_name", "Unknown")
        booking_ref = state.get("booking_ref", "UNKNOWN")
        amount = state.get("amount", self.amount)
        payment_method = state.get("payment_method", "")
        user_answer = state.get("user_answer", "").lower()
        
        logger.info(f"{self.name}: Processing payment of ₹{amount} for Bus {bus_number}")
        
        # If no payment method selected yet, ask for it
        if not payment_method:
            logger.info(f"{self.name}: Requesting payment method")
            return {
                "result": f"""💳 Payment Required: ₹{amount}

Select payment method:
1. Credit Card
2. Debit Card
3. UPI
4. Net Banking

Please enter your choice (1-4):""",
                "bus_number": bus_number,
                "booking_ref": booking_ref,
                "amount": amount,
                "agent": self.name,
                "status": "awaiting_payment_method"
            }
        
        # If payment method selected, confirm payment
        methods = {
            "1": "Credit Card",
            "2": "Debit Card",
            "3": "UPI",
            "4": "Net Banking",
            "credit": "Credit Card",
            "debit": "Debit Card",
            "upi": "UPI",
            "net": "Net Banking"
        }
        
        # Map user input to payment method
        selected_method = methods.get(user_answer, payment_method)
        
        # Generate transaction ID
        transaction_id = f"TXN{bus_number}{hash(passenger_name) % 100000:05d}"
        
        logger.info(f"{self.name}: Payment confirmed via {selected_method}")
        
        return {
            "result": f"""✅ Payment Processed Successfully!

💰 Amount: ₹{amount}
📌 Method: {selected_method}
🔑 Transaction ID: {transaction_id}
🎫 Booking Ref: {booking_ref}

Your ticket has been confirmed!""",
            "bus_number": bus_number,
            "booking_ref": booking_ref,
            "passenger_name": passenger_name,
            "amount": amount,
            "payment_method": selected_method,
            "transaction_id": transaction_id,
            "agent": self.name,
            "status": "payment_confirmed"
        }
    
    def confirm_payment(self, payment_method: Optional[str] = None) -> Dict:
        """
        Confirm payment received
        
        Args:
            payment_method: Payment method used (Card, UPI, etc.)
            
        Returns:
            Payment confirmation details
        """
        logger.info(f"{self.name}: Payment confirmed via {payment_method or 'default'}")
        
        return {
            "payment_status": "completed",
            "payment_method": payment_method or "Card",
            "timestamp": str(__import__('datetime').datetime.now())
        }


# ============================================
# CONFIRMATION AGENT
# ============================================

class ConfirmationAgent:
    """
    Agent responsible for final booking confirmation.
    Provides booking summary and ticket details.
    """
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.name = "ConfirmationAgent"
    
    def process(self, state: Dict) -> Dict:
        """
        Generate final booking confirmation
        
        Args:
            state: Current conversation state with booking details
            
        Returns:
            Final confirmation message
        """
        bus_number = state.get("bus_number", "Unknown")
        booking_ref = state.get("booking_ref", "UNKNOWN")
        passenger_name = state.get("passenger_name", "Passenger")
        amount = state.get("amount", 120)
        
        logger.info(f"{self.name}: Generating confirmation for {booking_ref}")
        
        result_message = f"""
🎉 Thank you for booking Bus {bus_number}!

📋 Booking Details:
- Booking Reference: {booking_ref}
- Passenger Name: {passenger_name}
- Bus Number: {bus_number}
- Amount Paid: ₹{amount}
- Status: CONFIRMED

✉️ A confirmation email will be sent shortly.
📲 Download your ticket from the app.
📞 Support: 1800-TRANSPORT

Safe travels! 🚌
        """
        
        return {
            "result": result_message,
            "bus_number": bus_number,
            "booking_ref": booking_ref,
            "agent": self.name,
            "status": "booking_confirmed"
        }
