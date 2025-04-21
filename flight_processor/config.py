DATABASE_PATH = "database/relational/flight_data.db"
OPENAI_VECTOR_PATH = "database/vector/flight_vectors_openai"
LLAMA_VECTOR_PATH = "database/vector/flight_vectors_llama"

LLM_MODEL = "gpt-3.5-turbo"
LLM_TEMPERATURE = 0

LLAMA_MODEL_PATH = "llm/llama-2-7b-chat.Q8_0.gguf"
LLAMA_MODEL_N_CTX = 4096

VECTOR_SEARCH_TOP_K = 3

EXAMPLE_QUERIES = {
    "All flights": "SELECT * FROM flights",
    "Flights by airline": "SELECT * FROM flights WHERE airline = 'QFA'",
    "Flights between airports": "SELECT * FROM flights WHERE departure_port = 'DUD' AND arrival_port = 'AKL'",
    "Flight count by aircraft type": "SELECT aircraft_type, COUNT(*) FROM flights GROUP BY aircraft_type",
    "Flights on a specific date": "SELECT * FROM flights WHERE origin_date_local = '2024-12-09'"
}

FLIGHTS_TABLE_SCHEMA = '''
                       CREATE TABLE IF NOT EXISTS flights (
                                                              id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                              airline TEXT,
                                                              airline2 TEXT,
                                                              flight_number TEXT,
                                                              origin_date_local TEXT,
                                                              origin_date_utc TEXT,
                                                              domain TEXT,
                                                              category TEXT,
                                                              departure_port TEXT,
                                                              departure_country TEXT,
                                                              departure_time TEXT,
                                                              arrival_port TEXT,
                                                              arrival_country TEXT,
                                                              arrival_time TEXT,
                                                              status TEXT,
                                                              aircraft_registration TEXT,
                                                              aircraft_type TEXT,
                                                              aircraft_owner_airline TEXT,
                                                              capacity INTEGER,
                                                              raw_data TEXT,
                                                              UNIQUE(airline, flight_number, origin_date_local, departure_port, arrival_port)
                       ) \
                       '''