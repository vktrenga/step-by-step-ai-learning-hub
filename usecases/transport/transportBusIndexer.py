import json
import chromadb
from sentence_transformers import SentenceTransformer

class TransportBusIndexer:
    def __init__(self, file_path="chennai_transport_dataset.json"):
        self.file_path = file_path
        # ✅ Persistent client with a folder path
        self.client = chromadb.PersistentClient(path="bus_data")
        self.data = self.read_transport_data()
        self.processed_data, self.metadata_map = self.process_data()
        self.embedded_data = self.embedding_data()
        self.store_data()

    def read_transport_data(self):
        with open(self.file_path, "r") as f:
            return json.load(f)

    def process_data(self):
        documents = []
        metadata_map = []
        for bus in self.data:
            bus_info = (
                f"Bus {bus['bus_number']} ({bus['bus_type']}) route {bus['route']}, "
                f"seats {bus['available_seats']} of {bus['total_seats']} available."
            )
            documents.append(bus_info)
            metadata_map.append({"bus_number": bus["bus_number"], "route": bus["route"]})

            for stop in bus["stops"]:
                stop_text = (
                    f"Stop {stop['stop_name']} arrival {stop['arrival_time']} "
                    f"departure {stop['departure_time']}."
                )
                if stop["fares"]:
                    fares_text = ", ".join(
                        [f"to {dest}: {fare} {bus['currency']}" for dest, fare in stop["fares"].items()]
                    )
                    stop_text += " Fares: " + fares_text
                documents.append(stop_text)
                metadata_map.append({"bus_number": bus["bus_number"], "route": bus["route"]})
        return documents, metadata_map

    def embedding_data(self):
        model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        embedded_data = model.encode(self.processed_data, convert_to_tensor=True)
        print("Shape of embeddings:", embedded_data.shape)
        return embedded_data

    def store_data(self):
        ids = [str(i) for i in range(len(self.processed_data))]
        documents = self.processed_data
        embeddings = self.embedded_data.tolist()
        metadatas = self.metadata_map

        print("Lengths:", len(ids), len(documents), len(embeddings), len(metadatas))

        # ✅ Create or get collection
        try:
            collection = self.client.get_collection("bus_data")
        except:
            collection = self.client.create_collection(name="bus_data")

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        # ✅ Persist to disk
        #self.client.persist()
        print("Data stored in pers  istent ChromaDB collection 'bus_data'.")

if __name__ == "__main__":
    TransportBusIndexer()
