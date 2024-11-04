import csv
from collections import namedtuple
import os
import random
import math
import logging
import requests

# Define data structures
Refinery = namedtuple(
    "Refinery", ["id", "capacity", "daily_output", "cost", "emissions"]
)
StorageTank = namedtuple(
    "StorageTank", ["id", "capacity", "intake_capacity", "outtake_capacity"]
)
Customer = namedtuple(
    "Customer", ["id", "demand", "delivery_start", "delivery_end", "penalty"]
)
TransportationChannel = namedtuple(
    "TransportationChannel",
    ["source", "destination", "type", "cost", "lead_time", "emissions"],
)


# Load data from CSV files
def load_data():
    base_path = "./eval-platform/src/main/resources/liquibase/data"

    refineries = []
    storage_tanks = []
    customers = []
    transportation_channels = []

    # Load refineries
    with open(os.path.join(base_path, "refineries.csv"), "r") as file:
        reader = csv.DictReader(file, delimiter=";")
        for row in reader:
            refineries.append(
                Refinery(
                    id=row["id"],
                    capacity=int(row["capacity"]),
                    daily_output=int(row["max_output"]),
                    cost=float(row["production_cost"]),
                    emissions=float(row["production_co2"]),
                )
            )

    # Load storage tanks
    with open(os.path.join(base_path, "tanks.csv"), "r") as file:
        reader = csv.DictReader(file, delimiter=";")
        for row in reader:
            storage_tanks.append(
                StorageTank(
                    id=row["id"],
                    capacity=int(row["capacity"]),
                    intake_capacity=int(row["max_input"]),
                    outtake_capacity=int(row["max_output"]),
                )
            )

    # Load demands to get delivery windows
    demands = {}
    with open(os.path.join(base_path, "demands.csv"), "r") as file:
        reader = csv.DictReader(file, delimiter=";")
        for row in reader:
            demands[row["customer_id"]] = {
                "demand": int(row["quantity"]),
                "delivery_start": int(row["start_delivery_day"]),
                "delivery_end": int(row["end_delivery_day"]),
            }

    # Load customers
    with open(os.path.join(base_path, "customers.csv"), "r") as file:
        reader = csv.DictReader(file, delimiter=";")
        for row in reader:
            customer_id = row["id"]
            if customer_id in demands:
                customers.append(
                    Customer(
                        id=customer_id,
                        demand=demands[customer_id]["demand"],
                        delivery_start=demands[customer_id]["delivery_start"],
                        delivery_end=demands[customer_id]["delivery_end"],
                        penalty=float(row["late_delivery_penalty"]),
                    )
                )

    # Load transportation channels
    with open(os.path.join(base_path, "connections.csv"), "r") as file:
        reader = csv.DictReader(file, delimiter=";")
        for row in reader:
            # Assuming emissions are calculated based on distance and a constant factor
            emissions_factor = 0.1  # Example factor, adjust as needed
            transportation_channels.append(
                TransportationChannel(
                    source=row["from_id"],
                    destination=row["to_id"],
                    type=row["connection_type"],
                    cost=float(row["distance"]),
                    lead_time=int(row["lead_time_days"]),
                    emissions=float(row["distance"]) * emissions_factor,
                )
            )

    return refineries, storage_tanks, customers, transportation_channels


def optimize_deliveries(refineries, storage_tanks, customers, transportation_channels):
    # Initialize variables to track movements and costs
    movements = []
    total_cost = 0
    total_emissions = 0

    # Define initial weights for multi-criteria decision making
    cost_weight = 0.5
    emissions_weight = 0.3
    capacity_weight = 0.2

    # Sort customers by a combination of delivery end day and penalty
    customers.sort(key=lambda c: (c.delivery_end, c.penalty))

    # Initial solution using heuristic
    for customer in customers:
        if not isinstance(customer, Customer):
            logging.error(f"Expected Customer object, got {type(customer)}")
            continue

        best_source = None
        best_score = float("inf")

        # Find the best source (refinery or storage tank) to fulfill the demand
        for source in refineries + storage_tanks:
            if source.capacity >= customer.demand:
                for channel in transportation_channels:
                    if (
                        channel.source == source.id
                        and channel.destination == customer.id
                    ):
                        cost = customer.demand * channel.cost
                        emissions = customer.demand * channel.emissions
                        lead_time = channel.lead_time

                        # Check if delivery can be made within the window
                        if lead_time <= (
                            customer.delivery_end - customer.delivery_start
                        ):
                            # Calculate a score based on multiple criteria
                            score = (
                                cost * cost_weight
                                + emissions * emissions_weight
                                + (source.capacity - customer.demand) * capacity_weight
                            )

                            # Check if this is the best option so far
                            if score < best_score:
                                best_source = source
                                best_score = score

        if best_source:
            # Record the movement
            movements.append((best_source.id, customer.id, customer.demand))
            total_cost += customer.demand * channel.cost
            total_emissions += customer.demand * channel.emissions
            # Create a new instance of the source with updated capacity
            if isinstance(best_source, Refinery):
                refineries = [
                    Refinery(
                        id=s.id,
                        capacity=(
                            s.capacity - customer.demand
                            if s.id == best_source.id
                            else s.capacity
                        ),
                        daily_output=s.daily_output,
                        cost=s.cost,
                        emissions=s.emissions,
                    )
                    for s in refineries
                ]
            elif isinstance(best_source, StorageTank):
                storage_tanks = [
                    StorageTank(
                        id=s.id,
                        capacity=(
                            s.capacity - customer.demand
                            if s.id == best_source.id
                            else s.capacity
                        ),
                        intake_capacity=s.intake_capacity,
                        outtake_capacity=s.outtake_capacity,
                    )
                    for s in storage_tanks
                ]

    # Simulated Annealing
    temperature = 100
    cooling_rate = 0.95
    while temperature > 1:
        # Generate a neighboring solution
        new_movements = movements[:]
        if new_movements:
            i = random.randint(0, len(new_movements) - 1)
            # Randomly adjust a movement
            new_movements[i] = (
                new_movements[i][0],
                new_movements[i][1],
                max(0, new_movements[i][2] + random.randint(-5, 5)),
            )

            # Calculate new cost and emissions
            new_total_cost = sum(
                m[2] * channel.cost
                for m in new_movements
                for channel in transportation_channels
                if channel.source == m[0] and channel.destination == m[1]
            )
            new_total_emissions = sum(
                m[2] * channel.emissions
                for m in new_movements
                for channel in transportation_channels
                if channel.source == m[0] and channel.destination == m[1]
            )

            # Calculate the change in cost and emissions
            delta_cost = new_total_cost - total_cost
            delta_emissions = new_total_emissions - total_emissions

            # Acceptance probability
            if (
                delta_cost < 0
                or delta_emissions < 0
                or random.random() < math.exp(-delta_cost / temperature)
            ):
                movements = new_movements
                total_cost = new_total_cost
                total_emissions = new_total_emissions

        # Decrease the temperature
        temperature *= cooling_rate

    # Return the planned movements, total cost, and emissions
    return movements, total_cost, total_emissions


# API interaction functions
def send_start_data(api_url_start, api_key):
    headers = {"API-KEY": f"{api_key}"}
    try:
        response = requests.post(api_url_start, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses
        logging.info("Session started successfully.")

        # The response is expected to be a plain string (session ID)
        session_id = response.text.strip()

        if session_id:
            logging.info(f"Session ID received: {session_id}")
        else:
            logging.warning("Session ID not found in the response.")

        return session_id
    except requests.exceptions.RequestException as e:
        logging.error(f"Error starting session: {e}")
        logging.debug(f"API Key: {api_key}\n")
        logging.debug(f"Headers: {headers}\n")
        return None


def send_end_data(api_url_end, api_key):
    headers = {"API-KEY": f"{api_key}"}
    try:
        response = requests.post(f"{api_url_end}", headers=headers)
        response.raise_for_status()  # Raise an error for bad responses
        logging.info("Delivery data sent successfully.")

        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error sending delivery data: {e}")
        logging.debug(f"API Key: {api_key}")
        logging.debug(f"Headers: {headers}\n\n\n\n\n")
        return None


def send_play_status(api_url, api_key, session_id, day, movements):
    headers = {
        "API-KEY": str(api_key),  # Ensure the API key is a string
        "SESSION-ID": str(session_id),  # Ensure the session ID is a string
        "Content-Type": "application/json",
    }
    # Prepare the payload with an empty movement for the first request
    delivery_data = {
        "day": day,
        "movements": []
    }
    try:
        response = requests.post(api_url, json=delivery_data, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses
        logging.info("Delivery status sent successfully.")

        # Process the response
        response_data = (
            response.json()
            if response.headers.get("Content-Type") == "application/json"
            else response.text.strip()
        )

        if response_data:
            logging.info(f"Response received: {response_data}")
        else:
            logging.warning("No response data found.")

        return response_data
    except requests.exceptions.RequestException as e:
        logging.error(f"Error sending delivery status: {e}")
        logging.debug(f"API Key: {api_key}\n")
        logging.debug(f"Headers: {headers}\n")
        return None


# Main simulation loop
def run_simulation():
    api_url_start = "http://localhost:8080/api/v1/session/start"
    api_url_end = "http://localhost:8080/api/v1/session/end"
    api_url_play = "http://localhost:8080/api/v1/play/round"

    with open("/Backend/API/api_key.txt", "r") as file:
        api_key = file.read().strip()

    session_id = send_start_data(api_url_start, api_key)

    try:
        # Read the API key from a file
        with open("/Backend/API/api_key.txt", "r") as file:
            api_key = file.read().strip()

        refineries, storage_tanks, customers, transportation_channels = load_data()
        for day in range(42):
            movements, total_cost, total_emissions = optimize_deliveries(
                refineries, storage_tanks, customers, transportation_channels
            ) # The function where deliveries are optimized, after the data is received from the API and before we send the results for scoring
            status = send_play_status(
                api_url_play, api_key, session_id, day, movements
            ) # The function where we send the results to the API for scoring
            logging.info(f"Day {day} completed with status: {status}")
    except Exception as e:
        logging.error(f"Error in simulation: {e}")

    send_end_data(api_url_end, api_key)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    run_simulation()
