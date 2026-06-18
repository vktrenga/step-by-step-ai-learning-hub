"""
Bus Data Parser and Formatter
Extracts and formats bus information correctly
"""

import json
import logging
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore", message="Accessing `__path__`")

logger = logging.getLogger(__name__)

class BusDataParser:
    """Parse and format bus data from context"""
    
    @staticmethod
    def extract_bus_info(context: str, bus_number: str, start_location: str = "", end_location: str = "") -> Dict:
        """
        Extract structured bus information from context
        
        Args:
            context: Raw context text containing bus data
            bus_number: Bus number to extract info for
            start_location: Starting location (optional, for intermediate stops)
            end_location: Ending location (optional, for intermediate stops)
            
        Returns:
            Dictionary with parsed bus info
        """
        try:
            # Try to parse as JSON first (if context is JSON-like)
            data = json.loads(context)
            
            if isinstance(data, list):
                # Find matching bus
                for bus in data:
                    if str(bus.get("bus_number")) == str(bus_number):
                        result = BusDataParser._format_bus_info(bus, start_location, end_location)
                        return result
            elif isinstance(data, dict):
                if str(data.get("bus_number")) == str(bus_number):
                    return BusDataParser._format_bus_info(data, start_location, end_location)
        except json.JSONDecodeError:
            # Not JSON, try text parsing
            logger.debug(f"Could not parse context as JSON for bus {bus_number}")
            pass
        
        # Fallback: extract from text
        return BusDataParser._extract_from_text(context, bus_number)
    
    @staticmethod
    def _format_bus_info(bus_data: Dict, start_location: str = "", end_location: str = "") -> Dict:
        """Format structured bus data from JSON"""
        stops = bus_data.get("stops", [])
        
        # Find departure time based on start location
        departure_time = stops[0].get("departure_time", "N/A") if stops else "N/A"
        if start_location:
            for stop in stops:
                if start_location.lower() in stop.get("stop_name", "").lower():
                    departure_time = stop.get("departure_time", departure_time)
                    break
        
        # Find price based on start->end location
        price = 0
        if start_location and end_location:
            for stop in stops:
                if start_location.lower() in stop.get("stop_name", "").lower():
                    fares = stop.get("fares", {})
                    for fare_dest, fare_price in fares.items():
                        if end_location.lower() in fare_dest.lower():
                            price = fare_price
                            break
                    break
        else:
            # Default: use first fare
            first_stop = stops[0] if stops else {}
            fares = first_stop.get("fares", {})
            price = list(fares.values())[0] if fares else 0
        
        # Extract route
        route_start = stops[0].get("stop_name", "Unknown") if stops else "Unknown"
        route_end = stops[-1].get("stop_name", "Unknown") if stops else "Unknown"
        
        return {
            "bus_number": str(bus_data.get("bus_number", "Unknown")),
            "route": f"{route_start} → {route_end}",
            "departure_time": departure_time,
            "price": int(price) if price else 0,
            "available_seats": bus_data.get("available_seats", 0),
            "total_seats": bus_data.get("total_seats", 0),
            "bus_type": bus_data.get("bus_type", "Standard"),
            "stops": stops,
            "currency": "INR"
        }
    
    @staticmethod
    def _extract_from_text(context: str, bus_number: str) -> Dict:
        """Extract bus info from unstructured text"""
        import re
        
        info = {
            "bus_number": bus_number,
            "route": "Unknown",
            "departure_time": "N/A",
            "price": 0,
            "available_seats": 0,
            "total_seats": 0,
            "currency": "INR"
        }
        
        # Try to extract route
        route_pattern = r'(?:Route|route|from|to):\s*([^,\n]+(?:→|->|to)\s*[^,\n]+)'
        route_match = re.search(route_pattern, context)
        if route_match:
            info["route"] = route_match.group(1).strip()
        
        # Try to extract time
        time_pattern = r'(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?'
        time_match = re.search(time_pattern, context)
        if time_match:
            info["departure_time"] = time_match.group(0)
        
        # Try to extract price (look for ₹ symbol)
        price_pattern = r'₹\s*(\d+(?:,\d{3})*)'
        price_match = re.search(price_pattern, context)
        if price_match:
            info["price"] = int(price_match.group(1).replace(",", ""))
        
        # Try to extract seats
        seats_pattern = r'(\d+)\s*seats?'
        seats_match = re.search(seats_pattern, context, re.IGNORECASE)
        if seats_match:
            info["available_seats"] = int(seats_match.group(1))
        
        return info
    
    @staticmethod
    def format_bus_response(bus_info: Dict) -> str:
        """
        Format bus information into response string
        
        Args:
            bus_info: Parsed bus information dictionary
            
        Returns:
            Formatted response string
        """
        bus_number = bus_info.get("bus_number", "Unknown")
        route = bus_info.get("route", "Unknown")
        departure_time = bus_info.get("departure_time", "N/A")
        price = bus_info.get("price", 0)
        available_seats = bus_info.get("available_seats", 0)
        bus_type = bus_info.get("bus_type", "Standard")
        
        formatted = (
            f"Bus {bus_number} | "
            f"Route: {route} | "
            f"Departure: {departure_time} | "
            f"Price: ₹{price} | "
            f"Type: {bus_type} | "
            f"Status: Available ({available_seats} seats)"
        )
        
        return formatted
    
    @staticmethod
    def format_multiple_buses(buses_info: List[Dict]) -> str:
        """Format multiple bus information"""
        lines = []
        for i, bus_info in enumerate(buses_info, 1):
            formatted = BusDataParser.format_bus_response(bus_info)
            lines.append(f"{i}. {formatted}")
        return "\n".join(lines)


class BusDataEnhancer:
    """Add additional context to bus information"""
    
    @staticmethod
    def add_recommendations(bus_info: Dict) -> str:
        """Add recommendations based on bus info"""
        recommendations = []
        
        bus_type = bus_info.get("bus_type", "").lower()
        available_seats = bus_info.get("available_seats", 0)
        price = bus_info.get("price", 0)
        
        # Seat availability recommendation
        if available_seats <= 5:
            recommendations.append("⚠️ Only few seats available!")
        elif available_seats > 20:
            recommendations.append("✅ Good availability")
        
        # Bus type recommendation
        if "ac" in bus_type or "electric" in bus_type:
            recommendations.append("🌬️ AC/Comfortable ride")
        
        # Price range recommendation
        if price < 100:
            recommendations.append("💰 Budget-friendly")
        elif price > 500:
            recommendations.append("💎 Premium service")
        
        return " | ".join(recommendations) if recommendations else ""
    
    @staticmethod
    def add_time_conversion(time_str: str) -> str:
        """Convert time to 12-hour format"""
        import re
        
        match = re.match(r'(\d{1,2}):(\d{2})', time_str)
        if match:
            hour = int(match.group(1))
            minute = match.group(2)
            
            period = "AM" if hour < 12 else "PM"
            if hour > 12:
                hour -= 12
            elif hour == 0:
                hour = 12
            
            return f"{hour}:{minute} {period}"
        
        return time_str


# Example usage in searchTransport.py
def improve_bus_response(context: str, bus_number: str) -> Tuple[str, Dict]:
    """
    Improve bus response with better formatting
    
    Args:
        context: Raw context from RAG
        bus_number: Bus number
        
    Returns:
        Tuple of (formatted response, bus info dict)
    """
    # Parse bus info
    bus_info = BusDataParser.extract_bus_info(context, bus_number)
    
    # Format response
    formatted = BusDataParser.format_bus_response(bus_info)
    
    # Add recommendations
    recommendations = BusDataEnhancer.add_recommendations(bus_info)
    
    # Combine
    if recommendations:
        response = f"{formatted}\n{recommendations}"
    else:
        response = formatted
    
    return response, bus_info
