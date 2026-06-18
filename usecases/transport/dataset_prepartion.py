import json
import random
from datetime import datetime, timedelta

# Define routes with 5–10 stops each, including opposite directions
routes = {
    # Forward
    "Kelambakkam → Kilambakkam": [
        "Kelambakkam", "Mambakkam", "Sholinganallur", "Medavakkam", "Tambaram", "Perungalathur", "Vandalur", "Kilambakkam"
    ],
    # Reverse
    "Kilambakkam → Kelambakkam": [
        "Kilambakkam", "Vandalur", "Perungalathur", "Tambaram", "Medavakkam", "Sholinganallur", "Mambakkam", "Kelambakkam"
    ],
    # Forward
    "Kilambakkam → Kovalam": [
        "Kilambakkam", "Vandalur", "Medavakkam", "Sholinganallur", "Thiruvanmiyur", "Adyar", "Kottivakkam", "Kovalam"
    ],
    # Reverse
    "Kovalam → Kilambakkam": [
        "Kovalam", "Kottivakkam", "Adyar", "Thiruvanmiyur", "Sholinganallur", "Medavakkam", "Vandalur", "Kilambakkam"
    ],
    # Forward
    "Tambaram → Chengalpattu": [
        "Tambaram", "Perungalathur", "Vandalur", "Urapakkam", "Guduvanchery", "Singaperumal Koil", "Chengalpattu"
    ],
    # Reverse
    "Chengalpattu → Tambaram": [
        "Chengalpattu", "Singaperumal Koil", "Guduvanchery", "Urapakkam", "Vandalur", "Perungalathur", "Tambaram"
    ],
    # Forward
    "Thiruvanmiyur → Velachery": [
        "Thiruvanmiyur", "Indira Nagar", "Taramani", "Perungudi", "Madipakkam", "Velachery"
    ],
    # Reverse
    "Velachery → Thiruvanmiyur": [
        "Velachery", "Madipakkam", "Perungudi", "Taramani", "Indira Nagar", "Thiruvanmiyur"
    ],
    # Forward
    "Guindy → T Nagar": [
        "Guindy", "Saidapet", "CIT Nagar", "Nandanam", "Teynampet", "T Nagar"
    ],
    # Reverse
    "T Nagar → Guindy": [
        "T Nagar", "Teynampet", "Nandanam", "CIT Nagar", "Saidapet", "Guindy"
    ],
    # Forward
    "Adyar → CMBT": [
        "Adyar", "Anna University", "Guindy", "Ashok Nagar", "Vadapalani", "Koyambedu (CMBT)"
    ],
    # Reverse
    "CMBT → Adyar": [
        "Koyambedu (CMBT)", "Vadapalani", "Ashok Nagar", "Guindy", "Anna University", "Adyar"
    ]
}

bus_types = ["AC Volvo", "Non-AC City Bus", "Sleeper", "Electric"]

def generate_bus_record(route_name, stops, bus_number):
    # Random departure time
    start_time = datetime.strptime("06:00", "%H:%M") + timedelta(minutes=random.randint(0, 600))
    total_seats = random.choice([40, 50])
    available_seats = random.randint(5, total_seats)

    # Build stop timings and fares
    stop_data = []
    current_time = start_time
    for i, stop in enumerate(stops):
        arrival = current_time.strftime("%H:%M")
        departure = (current_time + timedelta(minutes=5)).strftime("%H:%M")

        # Stage-based fares: increasing with distance
        fares = {}
        for j in range(i+1, len(stops)):
            fares[stops[j]] = random.randint(10*(j-i), 20*(j-i))

        stop_data.append({
            "stop_name": stop,
            "arrival_time": arrival,
            "departure_time": departure,
            "fares": fares if fares else None
        })

        # Increment time for next stop
        current_time += timedelta(minutes=random.randint(15, 25))

    return {
        "bus_number": str(bus_number),
        "route": route_name,
        "bus_type": random.choice(bus_types),
        "total_seats": total_seats,
        "available_seats": available_seats,
        "currency": "INR",
        "stops": stop_data
    }

# Generate dataset
dataset = []
bus_id = 200
for route_name, stops in routes.items():
    for i in range(50):  # 50 records per route
        dataset.append(generate_bus_record(route_name, stops, bus_id))
        bus_id += 1

# Save to JSON file
with open("chennai_transport_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)

print("Dataset generated: chennai_transport_dataset.json")
