import sqlite3
import streamlit as st
from config import DATABASE_PATH, FLIGHTS_TABLE_SCHEMA


def setup_database():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute(FLIGHTS_TABLE_SCHEMA)
    conn.commit()
    return conn


def get_db_connection():
    return sqlite3.connect(DATABASE_PATH)


def store_flight_data(data):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute('''
        INSERT OR REPLACE INTO flights
        (airline, airline2, flight_number, origin_date_local, origin_date_utc,
         domain, category, departure_port, departure_country, departure_time,
         arrival_port, arrival_country, arrival_time, status, aircraft_registration,
         aircraft_type, aircraft_owner_airline, capacity, raw_data)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data.get("airline"),
            data.get("airline2"),
            data.get("flight_number"),
            data.get("origin_date_local"),
            data.get("origin_date_utc"),
            data.get("domain"),
            data.get("category"),
            data.get("departure_port"),
            data.get("departure_country"),
            data.get("departure_time"),
            data.get("arrival_port"),
            data.get("arrival_country"),
            data.get("arrival_time"),
            data.get("status"),
            data.get("aircraft_registration"),
            data.get("aircraft_type"),
            data.get("aircraft_owner_airline"),
            data.get("capacity"),
            data.get("raw_data")
        ))

        conn.commit()
        return True

    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return False
    finally:
        conn.close()


def get_flight_count():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM flights")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_flight_sample(limit=10):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(f"""
    SELECT airline, flight_number, origin_date_local,
           departure_port, departure_time,
           arrival_port, arrival_time, status
    FROM flights LIMIT {limit}
    """)

    columns = [description[0] for description in cursor.description]
    data = cursor.fetchall()
    conn.close()
    return columns, data


def execute_query(query):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(query)

    columns = [description[0] for description in cursor.description]
    data = cursor.fetchall()
    conn.close()
    return columns, data


def get_flight_by_id(flight_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM flights WHERE id = ?", (flight_id,))

    columns = [description[0] for description in cursor.description]
    flight_data = cursor.fetchone()
    conn.close()
    return columns, flight_data


def clear_database():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM flights")
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Failed to clear database: {str(e)}")
        return False