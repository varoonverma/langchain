import os
import json
import xml.etree.ElementTree as ET
import streamlit as st
from langchain.chains import create_extraction_chain
from langchain.prompts import PromptTemplate
from persistence.models import ModelFactory


def parse_with_llm(xml_content, api_key=None, use_openai=False):
    if use_openai:
        return _parse_with_openai(xml_content, api_key)
    else:
        return _parse_atom_xml_directly(xml_content)


def _parse_with_openai(xml_content, api_key):
    if not api_key:
        st.error("OpenAI API key is required when using OpenAI")
        return {"error": "Missing API key", "raw_data": xml_content}

    os.environ["OPENAI_API_KEY"] = api_key
    llm = ModelFactory.get_llm(api_key, use_openai=True)

    schema = {
        "properties": {
            "airline": {"type": "string", "description": "The airline code (e.g., QFA)"},
            "airline2": {"type": "string", "description": "The alternative airline code (e.g., QF)"},
            "flight_number": {"type": "string", "description": "The flight number"},
            "origin_date_local": {"type": "string", "description": "The local date of origin"},
            "origin_date_utc": {"type": "string", "description": "The UTC date of origin"},
            "domain": {"type": "string", "description": "Flight domain (e.g., Domestic, International)"},
            "category": {"type": "string", "description": "Flight category from the Categories/Tag element"},
            "departure_port": {"type": "string", "description": "Departure airport code"},
            "departure_country": {"type": "string", "description": "Departure country code"},
            "departure_time": {"type": "string", "description": "Scheduled departure time"},
            "arrival_port": {"type": "string", "description": "Arrival airport code"},
            "arrival_country": {"type": "string", "description": "Arrival country code"},
            "arrival_time": {"type": "string", "description": "Scheduled arrival time"},
            "status": {"type": "string", "description": "Flight status (e.g., Planned)"},
            "aircraft_registration": {"type": "string", "description": "Aircraft registration number"},
            "aircraft_type": {"type": "string", "description": "Aircraft type code"},
            "aircraft_owner_airline": {"type": "string", "description": "Airline that owns the aircraft"},
            "capacity": {"type": "integer", "description": "Aircraft capacity"}
        },
        "required": ["airline", "flight_number", "departure_port", "arrival_port"]
    }

    try:
        chain = create_extraction_chain(schema, llm)
        simplified_xml = " ".join(xml_content.split())
        result = chain.run(simplified_xml)

        if isinstance(result, list) and len(result) > 0:
            extracted_data = result[0]
        else:
            extracted_data = result

        extracted_data["raw_data"] = xml_content
        return extracted_data

    except Exception as e:
        st.error(f"Error during extraction with OpenAI: {str(e)}")
        return _fallback_extraction_openai(llm, simplified_xml, xml_content)


def _parse_atom_xml_directly(xml_content):
    try:
        data = {"raw_data": xml_content}
        root = ET.fromstring(xml_content)
        ns = {'atom': 'urn://valence.aero/schemas/airtransport/ATOM/300'}

        flight_elem = root.find('.//atom:Flight', ns)
        if flight_elem is None:
            return {"error": "Flight element not found", "raw_data": xml_content}

        service = flight_elem.find('./atom:Service', ns)
        if service is not None:
            identifier = service.find('./atom:Identifier', ns)
            if identifier is not None:
                airline_elem = identifier.find('./atom:Airline', ns)
                data["airline"] = airline_elem.text if airline_elem is not None else None

                airline2_elem = identifier.find('./atom:Airline2', ns)
                data["airline2"] = airline2_elem.text if airline2_elem is not None else None

                flight_num_elem = identifier.find('./atom:FlightNumber', ns)
                data["flight_number"] = flight_num_elem.text if flight_num_elem is not None else None

                origin_date = identifier.find('./atom:OriginDate', ns)
                if origin_date is not None:
                    local_date = origin_date.find('./atom:Local', ns)
                    data["origin_date_local"] = local_date.text if local_date is not None else None

                    utc_date = origin_date.find('./atom:UTC', ns)
                    data["origin_date_utc"] = utc_date.text if utc_date is not None else None

            domain_elem = service.find('./atom:Domain', ns)
            data["domain"] = domain_elem.text if domain_elem is not None else None

            category_elem = service.find('./atom:Categories/atom:Tag', ns)
            data["category"] = category_elem.text if category_elem is not None else None

        leg = flight_elem.find('./atom:Leg', ns)
        if leg is not None:
            departure = leg.find('./atom:Departure', ns)
            if departure is not None:
                port_elem = departure.find('./atom:Port', ns)
                if port_elem is not None:
                    data["departure_port"] = port_elem.text
                    data["departure_country"] = port_elem.get('Country')

                schedule_elem = departure.find('./atom:Schedule', ns)
                data["departure_time"] = schedule_elem.text if schedule_elem is not None else None

            arrival = leg.find('./atom:Arrival', ns)
            if arrival is not None:
                port_elem = arrival.find('./atom:Port', ns)
                if port_elem is not None:
                    data["arrival_port"] = port_elem.text
                    data["arrival_country"] = port_elem.get('Country')

                schedule_elem = arrival.find('./atom:Schedule', ns)
                data["arrival_time"] = schedule_elem.text if schedule_elem is not None else None

            status_elem = leg.find('./atom:Status', ns)
            data["status"] = status_elem.text if status_elem is not None else None

            operation = leg.find('./atom:Operation', ns)
            if operation is not None:
                aircraft = operation.find('./atom:Aircraft', ns)
                if aircraft is not None:
                    reg_elem = aircraft.find('./atom:Registration', ns)
                    data["aircraft_registration"] = reg_elem.text if reg_elem is not None else None

                    type_elem = aircraft.find('./atom:Type', ns)
                    data["aircraft_type"] = type_elem.text if type_elem is not None else None

                    owner = aircraft.find('./atom:Owner', ns)
                    if owner is not None:
                        owner_airline = owner.find('./atom:Airline', ns)
                        data["aircraft_owner_airline"] = owner_airline.text if owner_airline is not None else None

                    try:
                        physical = aircraft.find('./atom:Configuration/atom:Cabin/atom:Physical', ns)
                        if physical is not None:
                            capacity_elem = physical.find('./atom:Capacity', ns)
                            data["capacity"] = int(capacity_elem.text) if capacity_elem is not None else None
                    except (ValueError, TypeError):
                        data["capacity"] = None

        required_fields = ["airline", "flight_number", "departure_port", "arrival_port"]
        missing_fields = [field for field in required_fields if field not in data or data[field] is None]

        if missing_fields:
            return _parse_atom_xml_without_namespaces(xml_content)

        return data

    except Exception:
        return _parse_atom_xml_without_namespaces(xml_content)


def _parse_atom_xml_without_namespaces(xml_content):
    try:
        data = {"raw_data": xml_content}
        root = ET.fromstring(xml_content)

        for elem in root.iter():
            if '}' in elem.tag:
                elem.tag = elem.tag.split('}')[-1]

        flight_elem = root.find(".//Flight")
        if flight_elem is None:
            return {"error": "Flight element not found", "raw_data": xml_content}

        service = flight_elem.find(".//Service")
        if service is not None:
            identifier = service.find(".//Identifier")
            if identifier is not None:
                airline_elem = identifier.find(".//Airline")
                data["airline"] = airline_elem.text if airline_elem is not None else None

                airline2_elem = identifier.find(".//Airline2")
                data["airline2"] = airline2_elem.text if airline2_elem is not None else None

                flight_num_elem = identifier.find(".//FlightNumber")
                data["flight_number"] = flight_num_elem.text if flight_num_elem is not None else None

                origin_date = identifier.find(".//OriginDate")
                if origin_date is not None:
                    local_date = origin_date.find(".//Local")
                    data["origin_date_local"] = local_date.text if local_date is not None else None

                    utc_date = origin_date.find(".//UTC")
                    data["origin_date_utc"] = utc_date.text if utc_date is not None else None

            domain_elem = service.find(".//Domain")
            data["domain"] = domain_elem.text if domain_elem is not None else None

            category_elem = service.find(".//Categories/Tag")
            data["category"] = category_elem.text if category_elem is not None else None

        leg = flight_elem.find(".//Leg")
        if leg is not None:
            departure = leg.find(".//Departure")
            if departure is not None:
                port_elem = departure.find(".//Port")
                if port_elem is not None:
                    data["departure_port"] = port_elem.text
                    data["departure_country"] = port_elem.get('Country')

                schedule_elem = departure.find(".//Schedule")
                data["departure_time"] = schedule_elem.text if schedule_elem is not None else None

            arrival = leg.find(".//Arrival")
            if arrival is not None:
                port_elem = arrival.find(".//Port")
                if port_elem is not None:
                    data["arrival_port"] = port_elem.text
                    data["arrival_country"] = port_elem.get('Country')

                schedule_elem = arrival.find(".//Schedule")
                data["arrival_time"] = schedule_elem.text if schedule_elem is not None else None

            status_elem = leg.find(".//Status")
            data["status"] = status_elem.text if status_elem is not None else None

            operation = leg.find(".//Operation")
            if operation is not None:
                aircraft = operation.find(".//Aircraft")
                if aircraft is not None:
                    reg_elem = aircraft.find(".//Registration")
                    data["aircraft_registration"] = reg_elem.text if reg_elem is not None else None

                    type_elem = aircraft.find(".//Type")
                    data["aircraft_type"] = type_elem.text if type_elem is not None else None

                    owner = aircraft.find(".//Owner")
                    if owner is not None:
                        owner_airline = owner.find(".//Airline")
                        data["aircraft_owner_airline"] = owner_airline.text if owner_airline is not None else None

                    try:
                        physical = aircraft.find(".//Configuration/Cabin/Physical")
                        if physical is not None:
                            capacity_elem = physical.find(".//Capacity")
                            if capacity_elem is not None and capacity_elem.text:
                                data["capacity"] = int(capacity_elem.text)
                    except (ValueError, TypeError):
                        data["capacity"] = None

        return data

    except Exception:
        flight_summary = _extract_flight_summary(xml_content)
        return {
            "airline": flight_summary.split()[1] if len(flight_summary.split()) > 1 else None,
            "flight_number": flight_summary.split()[2] if len(flight_summary.split()) > 2 else None,
            "raw_data": xml_content
        }


def _extract_flight_summary(xml_content):
    try:
        root = ET.fromstring(xml_content)
        for elem in root.iter():
            if '}' in elem.tag:
                elem.tag = elem.tag.split('}')[-1]

        flight_elem = root.find(".//Flight")
        service = flight_elem.find(".//Service")
        identifier = service.find(".//Identifier")

        airline = identifier.find(".//Airline").text
        flight_number = identifier.find(".//FlightNumber").text
        flight_summary = f"Flight {airline} {flight_number}"

        leg = flight_elem.find(".//Leg")
        if leg is not None:
            dep = leg.find(".//Departure/Port")
            arr = leg.find(".//Arrival/Port")
            if dep is not None and arr is not None:
                flight_summary += f" from {dep.text} to {arr.text}"

        return flight_summary

    except Exception:
        return "ATOM XML Flight Data"


def _fallback_extraction_openai(llm, simplified_xml, xml_content):
    flight_summary = _extract_flight_summary(xml_content)

    custom_prompt = PromptTemplate(
        input_variables=["xml", "summary"],
        template="""
        Extract structured information from this ATOM XML flight data for {summary}.
        
        XML:
        {xml}
        
        Extract the following as a JSON object:
        - airline: The airline code (e.g., QFA)
        - airline2: The alternative airline code (e.g., QF)
        - flight_number: The flight number
        - origin_date_local: The local date of origin
        - origin_date_utc: The UTC date of origin
        - departure_port: Departure airport code
        - departure_country: Departure country code
        - departure_time: Scheduled departure time
        - arrival_port: Arrival airport code
        - arrival_country: Arrival country code
        - arrival_time: Scheduled arrival time
        - status: Flight status
        - aircraft_registration: Aircraft registration number
        - aircraft_type: Aircraft type code
        
        Return ONLY a valid JSON object with no additional text.
        """
    )

    try:
        if hasattr(llm, 'invoke'):
            response = llm.invoke(custom_prompt.format(
                xml=simplified_xml[:10000],
                summary=flight_summary
            ))
            content = response.content if hasattr(response, 'content') else response
        else:
            content = llm(custom_prompt.format(
                xml=simplified_xml[:10000],
                summary=flight_summary
            ))

        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            extracted_data = json.loads(content[json_start:json_end])
            extracted_data["raw_data"] = xml_content
            return extracted_data
        else:
            return {"error": "Could not extract data", "raw_data": xml_content}

    except Exception:
        return {"error": "Extraction failed", "raw_data": xml_content}