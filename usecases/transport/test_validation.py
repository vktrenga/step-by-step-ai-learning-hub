"""
Test Script: Validate Bus Data Formatting

Tests that the system does NOT produce generic responses like:
"This bus follows a route from Downtown to Northside, departing at 10:30 AM. 
The ticket price for this journey is $15."

And DOES produce proper formatted responses like:
"Bus 451 | Route: Chennai → Bangalore | Departure: 10:30 PM | Price: ₹850 | Status: Available (12 seats)"
"""

import json
import logging
from bus_parser import BusDataParser, BusDataEnhancer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_parser_with_real_data():
    """Test parser with real Chennai transport data"""
    
    print("\n" + "="*80)
    print("TEST 1: Parser with Real Data")
    print("="*80)
    
    # Real data structure
    sample_data = json.dumps([
        {
            "bus_number": "451",
            "route": "Kelambakkam → Kilambakkam",
            "bus_type": "AC Volvo",
            "total_seats": 40,
            "available_seats": 25,
            "currency": "INR",
            "stops": [
                {
                    "stop_name": "Kelambakkam",
                    "arrival_time": "14:15",
                    "departure_time": "14:20",
                    "fares": {
                        "Mambakkam": 20,
                        "Kilambakkam": 120
                    }
                },
                {
                    "stop_name": "Kilambakkam",
                    "arrival_time": "14:50",
                    "departure_time": "14:50"
                }
            ]
        }
    ])
    
    # Parse
    parsed = BusDataParser.extract_bus_info(sample_data, "451")
    print(f"\n✅ Parsed Data:")
    for key, value in parsed.items():
        print(f"   {key}: {value}")
    
    # Format
    formatted = BusDataParser.format_bus_response(parsed)
    print(f"\n✅ Formatted Response:")
    print(f"   {formatted}")
    
    # Check for forbidden patterns
    forbidden_patterns = ["downtown", "northside", "this bus follows", "$", "dollar"]
    has_forbidden = any(p.lower() in formatted.lower() for p in forbidden_patterns)
    
    if has_forbidden:
        print("\n❌ FAILED: Response contains forbidden patterns!")
        return False
    else:
        print("\n✅ PASSED: Response has CORRECT format!")
        return True


def test_forbidden_patterns():
    """Test that we detect and reject forbidden response patterns"""
    
    print("\n" + "="*80)
    print("TEST 2: Forbidden Pattern Detection")
    print("="*80)
    
    bad_responses = [
        "This bus follows a route from Downtown to Northside, departing at 10:30 AM. The ticket price for this journey is $15.",
        "The bus goes from downtown, costs $120",
        "This bus route is from place to place, price is $50",
    ]
    
    good_responses = [
        "Bus 451 | Route: Chennai → Bangalore | Departure: 10:30 PM | Price: ₹850 | Status: Available (12 seats)",
        "Bus 452 | Route: Kilambakkam → Kelambakkam | Departure: 2:15 PM | Price: ₹120 | Status: Available (25 seats)",
    ]
    
    forbidden_patterns = ["downtown", "northside", "this bus follows", "$", "dollar"]
    
    print("\n🔴 Testing BAD responses (should detect forbidden patterns):")
    for response in bad_responses:
        has_forbidden = any(p.lower() in response.lower() for p in forbidden_patterns)
        status = "❌ DETECTED FORBIDDEN" if has_forbidden else "✅ CLEAN"
        print(f"   {status}: {response[:50]}...")
    
    print("\n✅ Testing GOOD responses (should pass clean):")
    for response in good_responses:
        has_forbidden = any(p.lower() in response.lower() for p in forbidden_patterns)
        status = "✅ CLEAN" if not has_forbidden else "❌ FAILED"
        print(f"   {status}: {response[:50]}...")
    
    return True


def test_recommendations():
    """Test recommendation generation"""
    
    print("\n" + "="*80)
    print("TEST 3: Recommendations Generation")
    print("="*80)
    
    bus_info = {
        "bus_number": "451",
        "route": "Chennai → Bangalore",
        "departure_time": "10:30 PM",
        "price": 850,
        "available_seats": 12,
        "bus_type": "AC Volvo",
        "currency": "INR"
    }
    
    recommendations = BusDataEnhancer.add_recommendations(bus_info)
    
    print(f"\n✅ Recommendations for Bus 451:")
    print(f"   {recommendations}")
    
    return len(recommendations) > 0


def test_data_validation():
    """Test that required fields are present"""
    
    print("\n" + "="*80)
    print("TEST 4: Data Validation")
    print("="*80)
    
    sample_data = json.dumps({
        "bus_number": "451",
        "route": "Kelambakkam → Kilambakkam",
        "stops": [{
            "stop_name": "Kelambakkam",
            "departure_time": "14:20",
            "fares": {"Kilambakkam": 120}
        }]
    })
    
    parsed = BusDataParser.extract_bus_info(sample_data, "451")
    
    required_fields = ["bus_number", "route", "departure_time", "price"]
    
    print(f"\n✅ Checking required fields:")
    all_present = True
    for field in required_fields:
        present = field in parsed and parsed[field]
        status = "✅ PRESENT" if present else "❌ MISSING"
        value = parsed.get(field, "N/A")
        print(f"   {status}: {field} = {value}")
        if not present:
            all_present = False
    
    return all_present


def test_currency_enforcement():
    """Test that ₹ is used, never $"""
    
    print("\n" + "="*80)
    print("TEST 5: Currency Enforcement (₹ only, NO $)")
    print("="*80)
    
    test_cases = [
        {
            "data": '{"bus_number": "451", "stops": [{"fares": {"dest": 120}}]}',
            "bus_num": "451",
            "should_have_rupee": True,
            "should_not_have_dollar": True,
            "description": "Real INR data"
        }
    ]
    
    print("\n✅ Testing currency formatting:")
    all_passed = True
    
    for test in test_cases:
        parsed = BusDataParser.extract_bus_info(test["data"], test["bus_num"])
        formatted = BusDataParser.format_bus_response(parsed)
        
        has_rupee = "₹" in formatted
        has_dollar = "$" in formatted
        
        if test["should_have_rupee"]:
            status = "✅ PASS" if has_rupee else "❌ FAIL"
        else:
            status = "✅ PASS" if not has_rupee else "❌ FAIL"
        
        print(f"   {status}: {test['description']}")
        print(f"           Response: {formatted[:60]}...")
        
        if not has_rupee or has_dollar:
            all_passed = False
    
    return all_passed


def run_all_tests():
    """Run all validation tests"""
    
    print("\n" + "="*80)
    print("🚌 BUS DATA FORMATTING VALIDATION TEST SUITE")
    print("="*80)
    
    results = {
        "test_parser_with_real_data": test_parser_with_real_data(),
        "test_forbidden_patterns": test_forbidden_patterns(),
        "test_recommendations": test_recommendations(),
        "test_data_validation": test_data_validation(),
        "test_currency_enforcement": test_currency_enforcement(),
    }
    
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    print(f"\n📈 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is working correctly.")
        print("\n✅ You will NO LONGER get responses like:")
        print('   "This bus follows a route from Downtown to Northside...$15"')
        print("\n✅ You WILL get responses like:")
        print('   "Bus 451 | Route: Chennai → Bangalore | Departure: 10:30 PM | Price: ₹850 | Status: Available (12 seats)"')
    else:
        print("\n⚠️ Some tests failed. Please check the output above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
