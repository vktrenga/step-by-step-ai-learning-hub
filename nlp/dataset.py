import pandas as pd
import random

# Tamil Nadu cities with lat/lon
cities = {
    "Chennai": (13.0827, 80.2707),
    "Coimbatore": (11.0168, 76.9558),
    "Trichy": (10.7905, 78.7047),
    "Madurai": (9.9252, 78.1198),
    "Salem": (11.6643, 78.1460)
}

brands = ["Samsung", "Apple", "Dell", "Lenovo", "HP", "Xiaomi", "OnePlus"]
categories = ["Mobile", "Laptop", "Electronics"]
reviews = [
    "Excellent performance and battery life.",
    "Value for money, highly recommended.",
    "Average quality, expected better.",
    "Fast delivery and good packaging.",
    "Seller was very responsive and helpful."
]

data = []
for i in range(10000):
    city, (lat, lon) = random.choice(list(cities.items()))
    record = {
        "product_id": f"P{i+1}",
        "product_name": f"{random.choice(brands)} {random.choice(categories)} {random.randint(1,100)}",
        "category": random.choice(categories),
        "price": random.randint(10000, 80000),
        "brand": random.choice(brands),
        "review_text": random.choice(reviews),
        "review_rating": round(random.uniform(1,5),1),
        "seller_id": f"S{random.randint(1000,9999)}",
        "seller_name": f"{city} Electronics Hub {random.randint(1,50)}",
        "seller_location": {"city": city, "lat": lat, "lon": lon},
        "seller_rating": round(random.uniform(3,5),1)
    }
    data.append(record)

# Save to JSON
pd.DataFrame(data).to_json("electronics_tn.json", orient="records", lines=True)

# Save to CSV
# pd.DataFrame(data).to_csv("electronics_tn.csv", index=False)
