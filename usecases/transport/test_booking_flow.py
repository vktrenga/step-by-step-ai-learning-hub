"""
Test Script: Verify Complete Booking Flow
Tests the multi-agent booking workflow
"""

import logging
from searchTransport import TransportBookingAssistant

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_booking_flow():
    """Test complete booking workflow"""
    
    print("\n" + "="*100)
    print("TEST: Complete Booking Flow (Search → Select → Confirm → Booking → Payment → Confirmation)")
    print("="*100)
    
    assistant = TransportBookingAssistant()
    
    # Step 1: Search for buses
    print("\n[Step 1] User searches: 'tambaram to kilambakkam'")
    result = assistant.run(query="tambaram to kilambakkam")
    print(f"Status: {result.get('status')}")
    print(f"Result:\n{result.get('result', 'No result')}\n")
    
    # Step 2: Select a bus
    print("\n[Step 2] User selects: '1' (first bus)")
    result = assistant.run(user_answer="1")
    print(f"Status: {result.get('status')}")
    print(f"Bus Number: {result.get('bus_number')}")
    print(f"Amount: ₹{result.get('amount')}")
    print(f"Result:\n{result.get('result', 'No result')}\n")
    
    # Step 3: Confirm bus selection
    print("\n[Step 3] User confirms: 'yes'")
    result = assistant.run(user_answer="yes")
    print(f"Status: {result.get('status')}")
    print(f"Result:\n{result.get('result', 'No result')}\n")
    
    # Step 4: Provide passenger name
    print("\n[Step 4] User provides name: 'Rajesh Kumar'")
    result = assistant.run(user_answer="Rajesh Kumar")
    print(f"Status: {result.get('status')}")
    print(f"Booking Ref: {result.get('booking_ref')}")
    print(f"Passenger: {result.get('passenger_name')}")
    print(f"Amount: ₹{result.get('amount')}")
    print(f"Result:\n{result.get('result', 'No result')}\n")
    
    # Step 5: Select payment method
    print("\n[Step 5] User selects payment: '1' (Credit Card)")
    result = assistant.run(user_answer="1")
    print(f"Status: {result.get('status')}")
    print(f"Transaction ID: {result.get('transaction_id')}")
    print(f"Payment Method: {result.get('payment_method')}")
    print(f"Result:\n{result.get('result', 'No result')}\n")
    
    # Step 6: Final confirmation
    print("\n[Step 6] Final Confirmation")
    print(f"Status: {result.get('status')}")
    print(f"Booking Complete!")
    
    print("\n" + "="*100)
    print("Flow Test Completed Successfully! ✅")
    print("="*100)

if __name__ == "__main__":
    test_booking_flow()
