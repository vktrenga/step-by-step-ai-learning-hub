import logging
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END
from prompts import SEARCH_TEMPLATE
from agents import SelectionAgent, ConfirmBusAgent, BookingAgent, PaymentAgent, ConfirmationAgent
from bus_parser import BusDataParser, BusDataEnhancer
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore", message="Accessing `__path__`")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransportBookingAssistant:
    def __init__(self, persist_dir="bus_data"):
        self.retriever = Chroma(
            persist_directory=persist_dir,
            collection_name="bus_data"
        ).as_retriever()

        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")

        self.llm = ChatOpenAI(model="gpt-4", temperature=0, max_tokens=200, verbose=False, api_key=self.api_key)
        self.graph = StateGraph(dict)
        
        # Initialize specialized agents
        self.selection_agent = SelectionAgent(self.llm)
        self.confirm_agent = ConfirmBusAgent(self.llm)
        self.booking_agent = BookingAgent(self.llm)
        self.payment_agent = PaymentAgent(self.llm)
        self.confirmation_agent = ConfirmationAgent(self.llm)
        
        logger.info("TransportBookingAssistant initialized with specialized agents")
        
        self._build_graph()
        self.app = self.graph.compile()
        self.state = None
        self.current_node = None
        self.execution_state = None

    def search_bus(self, state: dict):
        query = state.get("query", "").lower()
        logger.info(f"search_bus: Processing query: {query}")
        
        # Get all documents
        docs = self.retriever.invoke(query)
        logger.info(f"search_bus: Retrieved {len(docs)} documents for query: {query}")
        
        # Extract location names from query (e.g., "tambaram to kilambakkam" → ["tambaram", "kilambakkam"])
        locations = [loc.strip() for loc in query.replace("to", ",").split(",")]
        locations = [loc for loc in locations if loc]
        start_loc = locations[0] if locations else ""
        end_loc = locations[-1] if len(locations) > 1 else ""
        
        logger.info(f"search_bus: Extracted locations - Start: {start_loc}, End: {end_loc}")
        
        if not docs:
            logger.warning(f"search_bus: No buses found for query: {query}")
            return {"result": "No bus info found.", "available_buses": [], "bus_list": ""}

        bus_numbers = [d.metadata.get("bus_number", "Unknown") for d in docs if d.metadata]
        bus_numbers = list(set(bus_numbers))

        # Parse info for ALL buses
        all_buses_info = []
        bus_list_text = f"🚌 Found {len(bus_numbers)} matching buses:\n\n"
        
        for idx, bus_num in enumerate(bus_numbers, 1):
            # Filter docs for this specific bus
            bus_docs = [d for d in docs if d.metadata.get("bus_number") == bus_num]
            context = "\n".join([d.page_content for d in bus_docs])
            
            # Pass location info to parser
            parsed_bus_info = BusDataParser.extract_bus_info(context, bus_num, start_loc, end_loc)
            logger.info(f"search_bus: Parsed info for Bus {bus_num} - {parsed_bus_info}")
            
            # Format this bus's info
            has_any_data = (
                parsed_bus_info and (
                    parsed_bus_info.get("price") or
                    parsed_bus_info.get("route") or
                    parsed_bus_info.get("departure_time")
                )
            )

            if has_any_data:
                formatted_bus_response = BusDataParser.format_bus_response(parsed_bus_info)
                
                # Add placeholders for missing fields
                missing_fields = []
                if not parsed_bus_info.get("price"):
                    missing_fields.append("Price not available")
                if not parsed_bus_info.get("route"):
                    missing_fields.append("Route not available")
                if not parsed_bus_info.get("departure_time"):
                    missing_fields.append("Departure time not available")

                if missing_fields:
                    formatted_bus_response += "\n" + "; ".join(missing_fields)
            else:
                formatted_bus_response = f"Bus {bus_num} - Details not available"

            # Add to list display
            bus_list_text += f"{idx}. {formatted_bus_response}\n\n"
            
            all_buses_info.append({
                "bus_number": bus_num,
                "bus_details": formatted_bus_response,
                "parsed_info": parsed_bus_info,
                "index": idx
            })

        bus_list_text += f"👉 Please select a bus (enter the number 1-{len(bus_numbers)})"
        
        result = {
            "result": bus_list_text,
            "available_buses": all_buses_info,
            "bus_list": bus_list_text,
            "bus_count": len(bus_numbers),
            "data_source": "parser",
            "query_locations": {"start": start_loc, "end": end_loc}
        }
        
        logger.info(f"search_bus: Listed {len(bus_numbers)} available buses")
        return result

    def select_bus(self, state: dict):
        """Delegate to SelectionAgent for bus selection from list"""
        logger.info("select_bus: Delegating to SelectionAgent")
        result = self.selection_agent.process(state)
        
        # If bus was selected, extract the price from parsed_info
        if result.get("status") == "selection_complete":
            parsed_info = result.get("parsed_info", {})
            price = parsed_info.get("price", 120)
            result["amount"] = price
            logger.info(f"select_bus: Amount set to ₹{price}")
        
        return result

    def confirm_bus(self, state: dict):
        """Delegate to ConfirmBusAgent for bus confirmation"""
        logger.info("confirm_bus: Delegating to ConfirmBusAgent")
        result = self.confirm_agent.process(state)
        
        # Preserve amount in result
        if state.get("amount"):
            result["amount"] = state.get("amount")
        
        return result

    def booking_intent(self, state: dict):
        """Delegate to BookingAgent"""
        logger.info("booking_intent: Delegating to BookingAgent")
        result = self.booking_agent.process(state)
        
        # Preserve important fields
        result["amount"] = state.get("amount", result.get("amount", 120))
        result["available_buses"] = state.get("available_buses", [])
        
        return result

    def payment(self, state: dict):
        """Delegate to PaymentAgent"""
        logger.info("payment: Delegating to PaymentAgent")
        result = self.payment_agent.process(state)
        
        # Preserve important fields
        result["available_buses"] = state.get("available_buses", [])
        
        return result

    def confirmation(self, state: dict):
        """Delegate to ConfirmationAgent"""
        logger.info("confirmation: Delegating to ConfirmationAgent")
        return self.confirmation_agent.process(state)

    def _select_choice(self, state: dict):
        """Route decision based on bus selection result"""
        status = state.get("status")
        
        # If selection valid, move to confirmation
        if status == "selection_complete":
            logger.info("_select_choice: Bus selected - proceeding to confirmation")
            return "confirm_bus"
        
        # Otherwise stay in selection
        logger.info("_select_choice: Invalid selection - staying in select_bus")
        return "select_bus"

    def _confirm_choice(self, state: dict):
        """Route decision based on confirmation"""
        # Check if user confirmed with yes/no
        if self.confirm_agent.should_proceed(state.get("user_answer", "")):
            logger.info("_confirm_choice: User confirmed - proceeding to booking")
            return "booking_intent"
        else:
            logger.info("_confirm_choice: User rejected - ending flow")
            return END

    def _booking_choice(self, state: dict):
        """Route decision based on booking agent"""
        if self.booking_agent.should_proceed_to_payment(state.get("user_answer", "")):
            logger.info("_booking_choice: User confirmed booking - proceeding to payment")
            return "payment"
        else:
            logger.info("_booking_choice: User cancelled booking - ending flow")
            return END

    def _build_graph(self):
        self.graph.add_node("search_bus", self.search_bus)
        self.graph.add_node("select_bus", self.select_bus)
        self.graph.add_node("confirm_bus", self.confirm_bus)
        self.graph.add_node("booking_intent", self.booking_intent)
        self.graph.add_node("payment", self.payment)
        self.graph.add_node("confirmation", self.confirmation)

        self.graph.set_entry_point("search_bus")
        self.graph.add_edge("search_bus", "select_bus")
        self.graph.add_conditional_edges("select_bus", self._select_choice)
        self.graph.add_conditional_edges("confirm_bus", self._confirm_choice)
        self.graph.add_conditional_edges("booking_intent", self._booking_choice)
        self.graph.add_edge("payment", "confirmation")
        self.graph.add_edge("confirmation", END)

    def run(self, query: str = "", user_answer: str = "", bus_number: int = 0):
        # Initialize state on first call
        if self.state is None:
            self.state = {"query": query, "user_answer": user_answer}
            self.current_node = "search_bus"
            logger.info(f"run: New conversation started with query: {query}")
        else:
            # On repeat calls, only update the answer
            if query:  # New query restarts the flow
                self.state = {"query": query, "user_answer": ""}
                self.current_node = "search_bus"
                logger.info(f"run: New query - restarting flow: {query}")
            elif user_answer:  # Just an answer, continue from current node
                self.state["user_answer"] = user_answer
                logger.info(f"run: Processing user answer: {user_answer} at node {self.current_node}")
                
                # Route based on current node
                if self.current_node == "select_bus":
                    self.state.update(self._execute_node("select_bus"))
                    status = self.state.get("status")
                    
                    if status == "selection_complete":
                        # Valid selection made, move to confirmation
                        self.current_node = "confirm_bus"
                        self.state["user_answer"] = ""  # Clear for confirmation step
                        self.state.update(self._execute_node("confirm_bus"))
                    else:
                        # Invalid selection, stay in select_bus
                        logger.info("run: Waiting for valid bus selection")
                    return self.state
                
                elif self.current_node == "confirm_bus":
                    # User responded to confirmation with yes/no
                    # Route based on their answer, don't re-execute confirm_bus
                    next_node = self._confirm_choice(self.state)
                    
                    if next_node != END:
                        self.current_node = next_node
                        self.state["user_answer"] = ""  # Clear for next step
                        self.state.update(self._execute_node(next_node))
                    else:
                        logger.info("run: User cancelled booking")
                    
                    return self.state
                
                elif self.current_node == "booking_intent":
                    status = self.state.get("status")
                    
                    # If awaiting passenger info, capture the name
                    if status == "awaiting_passenger_info":
                        self.state["passenger_name"] = user_answer
                        self.state["user_answer"] = ""
                        logger.info(f"run: Passenger name captured: {user_answer}")
                    
                    # Execute booking node
                    self.state.update(self._execute_node("booking_intent"))
                    status = self.state.get("status")
                    
                    if status == "booking_confirmed":
                        # Booking confirmed, move to payment
                        self.current_node = "payment"
                        self.state["user_answer"] = ""
                        self.state.update(self._execute_node("payment"))
                    
                    return self.state
                
                elif self.current_node == "payment":
                    status = self.state.get("status")
                    
                    # If awaiting payment method, capture it
                    if status == "awaiting_payment_method":
                        self.state["payment_method"] = user_answer
                        self.state["user_answer"] = ""
                        logger.info(f"run: Payment method: {user_answer}")
                    
                    # Execute payment node
                    self.state.update(self._execute_node("payment"))
                    status = self.state.get("status")
                    
                    if status == "payment_confirmed":
                        # Payment confirmed, move to confirmation
                        self.current_node = "confirmation"
                        self.state.update(self._execute_node("confirmation"))
                    
                    return self.state
                
                elif self.current_node == "confirmation":
                    logger.info("run: Order confirmed")
                    return self.state
                
                return self.state

        # First execution: run search_bus and wait for user input
        if self.current_node == "search_bus":
            logger.info("run: Executing search_bus node")
            self.state.update(self._execute_node("search_bus"))
            self.current_node = "select_bus"
            logger.info("run: Waiting for bus selection")

        return self.state

    def _execute_node(self, node_name: str):
        """Execute a single node and return its output"""
        logger.info(f"_execute_node: Executing {node_name}")
        
        if node_name == "search_bus":
            return self.search_bus(self.state)
        elif node_name == "select_bus":
            return self.select_bus(self.state)
        elif node_name == "confirm_bus":
            return self.confirm_bus(self.state)
        elif node_name == "booking_intent":
            return self.booking_intent(self.state)
        elif node_name == "payment":
            return self.payment(self.state)
        elif node_name == "confirmation":
            return self.confirmation(self.state)
        else:
            logger.error(f"_execute_node: Unknown node {node_name}")
            return {}
