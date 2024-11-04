################
# Define Imports
################
import pandas as pd
from typing import *
from enum import Enum
import logging
import requests
from collections import namedtuple
import uuid
import random
from datetime import datetime, timedelta
import json
from pprint import pp
import math


##################
# Define Constants
##################
ROOT_PATH = "./Backend/challenge/eval-platform/src/main/resources/liquibase/data"
CONNECTIONS_PATH = ROOT_PATH + "/connections.csv"
CUSTOMERS_PATH = ROOT_PATH + "/customers.csv"
DEMANDS_PATH = ROOT_PATH + "/demands.csv"
REFINERIES_PATH = ROOT_PATH + "/refineries.csv"
TANKS_PATH = ROOT_PATH + "/tanks.csv"
TEAMS_PATH = ROOT_PATH + "/teams.csv"


################
# Define Classes
################
class ConnectionType(Enum):
    PIPELINE = 1
    TRUCK = 2


class NodeType(Enum):
    REFINERY = (1,)
    STORAGE_TANK = 2
    CUSTOMER = 3


class Connection:
    def __init__(
        self,
        id,
        from_id,
        to_id,
        distance,
        lead_time_days,
        connection_type: ConnectionType,
        max_capacity,
    ):
        self.id = id
        self.from_id = from_id
        self.to_id = to_id
        self.distance = distance
        self.lead_time_days = lead_time_days
        self.connection_type = connection_type
        self.max_capacity = max_capacity


class Node:
    def __init__(self, neighbours: "List[Node]", connections: List[Connection]):
        self.neighbours = neighbours
        self.connections = connections

    def add_neighbour(self, neighbour: "Node", connection: Connection):
        self.neighbours.append(neighbour)
        self.connections.append(connection)


class Refinery(Node):
    def __init__(
        self,
        id,
        name,
        capacity,
        max_output,
        production,
        overflow_penalty,
        underflow_penalty,
        over_output_penalty,
        production_cost,
        production_co2,
        initial_stock,
        node_type: NodeType,
    ):
        super().__init__(list(), list())
        self.id = id
        self.name = name
        self.capacity = capacity
        self.max_output = max_output
        self.production = production
        self.overflow_penalty = overflow_penalty
        self.underflow_penalty = underflow_penalty
        self.over_output_penalty = over_output_penalty
        self.production_cost = production_cost
        self.production_co2 = production_co2
        self.initial_stock = initial_stock
        self.stock = initial_stock
        self.node_type = node_type

        assert (
            self.node_type == NodeType.REFINERY
        ), "The node type is not NodeType.REFINERY for a refinery"


class StorageTank(Node):
    def __init__(
        self,
        id,
        name,
        capacity,
        max_input,
        max_output,
        overflow_penalty,
        underflow_penalty,
        over_input_penalty,
        over_output_penalty,
        initial_stock,
        node_type: NodeType,
    ):
        super().__init__(list(), list())
        self.id = id
        self.name = name
        self.capacity = capacity
        self.max_input = max_input
        self.max_output = max_output
        self.overflow_penalty = overflow_penalty
        self.underflow_penalty = underflow_penalty
        self.over_input_penalty = over_input_penalty
        self.over_output_penalty = over_output_penalty
        self.initial_stock = initial_stock
        self.stock = initial_stock
        self.node_type = node_type

        assert (
            self.node_type == NodeType.STORAGE_TANK
        ), "The node type is not NodeType.STORAGE_TANK for a storage tank"


class Customer(Node):
    def __init__(
        self,
        id,
        name,
        max_input,
        over_input_penalty,
        late_delivery_penalty,
        early_delivery_penalty,
        node_type: NodeType,
    ):
        super().__init__(list(), list())
        self.id = id
        self.name = name
        self.max_input = max_input
        self.over_input_penalty = over_input_penalty
        self.late_delivery_penalty = late_delivery_penalty
        self.early_delivery_penalty = early_delivery_penalty
        self.node_type = node_type
        self.stock = 0

        assert (
            self.node_type == NodeType.CUSTOMER
        ), "The node type is not NodeType.CUSTOMER for a customer"


class Demand(Node):
    def __init__(
        self, id, customer_id, quantity, post_day, start_delivery_day, end_delivery_day
    ):
        super().__init__(list(), list())
        self.id = id
        self.customer_id = customer_id
        self.quantity = quantity
        self.post_day = post_day
        self.start_delivery_day = start_delivery_day
        self.end_delivery_day = end_delivery_day
        self.remaining_quantity = quantity


###############
# Convert Nodes
###############
def fromEnumToString(node_type: NodeType):
    match node_type:
        case NodeType.CUSTOMER:
            return "CUSTOMER"
        case NodeType.REFINERY:
            return "REFINERY"
        case NodeType.STORAGE_TANK:
            return "STORAGE_TANK"


########################
# Extract Data from CVSs
########################
class Extractor:
    # Storage for our API data
    def __init__(self):
        self.data_store = {}

    # Read the CSV file into a DataFrame
    def extract_csv(self, file_path, delimeter=";"):
        try:
            df = pd.read_csv(file_path, delimiter=delimeter)
            print("Data successfully loaded into a DataFrame.")

            return df
        except FileNotFoundError:
            print("Error: The file was not found. Please check the file path.")
        except pd.errors.EmptyDataError:
            print("Error: The file is empty. Please check the file content.")
        except pd.errors.ParserError:
            print(
                "Error: There was an issue with parsing the file. Please check the file format."
            )

        return None

    # Retrieve data from the CSV files
    def parse_csv_info(self, root_path=ROOT_PATH):
        connections_path = root_path + "/connections.csv"
        customers_path = root_path + "/customers.csv"
        demands_path = root_path + "/demands.csv"
        refineries_path = root_path + "/refineries.csv"
        tanks_path = root_path + "/tanks.csv"
        teams_path = root_path + "/teams.csv"

        self.data_store["connections_data"] = self.extract_csv(connections_path)
        self.data_store["customers_data"] = self.extract_csv(customers_path)
        self.data_store["demands_data"] = self.extract_csv(demands_path)
        self.data_store["refineries_data"] = self.extract_csv(refineries_path)
        self.data_store["tanks_data"] = self.extract_csv(tanks_path)
        self.data_store["teams_data"] = self.extract_csv(teams_path)

    def process_refineries(self):
        for _, row in self.data_store["refineries_data"].iterrows():
            if row.node_type == "REFINERY":
                row.node_type = NodeType.REFINERY
            else:
                row.node_type = NodeType.STORAGE_TANK

            new_refinery = Refinery(
                row.id,
                row.name,
                row.capacity,
                row.max_output,
                row.production,
                row.overflow_penalty,
                row.underflow_penalty,
                row.over_output_penalty,
                row.production_cost,
                row.production_co2,
                row.initial_stock,
                row.node_type,
            )
            self.data_store["nodes_map"][row.id] = new_refinery

    def process_tanks(self):
        for _, row in self.data_store["tanks_data"].iterrows():
            if row.node_type == "STORAGE_TANK":
                row.node_type = NodeType.STORAGE_TANK
            else:
                row.node_type = NodeType.REFINERY

            new_tank = StorageTank(
                row.id,
                row.name,
                row.capacity,
                row.max_input,
                row.max_output,
                row.overflow_penalty,
                row.underflow_penalty,
                row.over_input_penalty,
                row.over_output_penalty,
                row.initial_stock,
                row.node_type,
            )
            self.data_store["nodes_map"][row.id] = new_tank

    def process_customers(self):
        for _, row in self.data_store["customers_data"].iterrows():
            if row.node_type == "CUSTOMER":
                row.node_type = NodeType.CUSTOMER
            else:
                row.node_type = NodeType.REFINERY

            new_refinery = Customer(
                row.id,
                row.name,
                row.max_input,
                row.over_input_penalty,
                row.late_delivery_penalty,
                row.early_delivery_penalty,
                row.node_type,
            )
            self.data_store["nodes_map"][row.id] = new_refinery

    def process_connections(self):
        self.data_store["connections_map"] = {}
        self.data_store["edge_list"] = []
        self.data_store["json_list"] = []

        for _, row in self.data_store["connections_data"].iterrows():
            new_connection = Connection(
                row.id,
                row.from_id,
                row.to_id,
                row.distance,
                row.lead_time_days,
                row.connection_type,
                row.max_capacity,
            )
            self.data_store["connections_map"][row.id] = new_connection
            self.data_store["nodes_map"][row.from_id].add_neighbour(
                self.data_store["nodes_map"][row.to_id], new_connection
            )
            self.data_store["edge_list"].append((row.from_id, row.to_id))
            self.data_store["json_list"].append(
                (
                    {
                        "from_id": row.from_id,
                        "from_node_type": fromEnumToString(
                            self.data_store["nodes_map"][row.from_id].node_type
                        ),
                        "to_id": row.to_id,
                        "to_node_type": fromEnumToString(
                            self.data_store["nodes_map"][row.to_id].node_type
                        ),
                    }
                )
            )

    def process_demands(self):
        for _, row in self.data_store["demands_data"].iterrows():
            new_demand = Demand(
                row.id,
                row.customer_id,
                row.quantity,
                row.post_day,
                row.start_delivery_day,
                row.end_delivery_day,
            )
            self.data_store["demands"].append(new_demand)

    def process_everything(self):
        self.data_store["nodes_map"] = {}
        self.data_store["demands"] = []

        self.process_refineries()
        self.process_tanks()
        self.process_customers()
        self.process_connections()

        self.process_demands()

    def post_connection_info(self) -> str:
        json_string = json.dumps(self.data_store["json_list"], indent=4)
        return json_string


class SolutionAPI:
    BASE_URL = "http://localhost:8080/api/v1"
    START_URL = BASE_URL + "/session/start"
    ROUND_URL = BASE_URL + "/play/round"
    END_URL = BASE_URL + "/session/end"
    API_KEY = "7bcd6334-bc2e-4cbf-b9d4-61cb9e868869"
    NUMBER_OF_DAYS = 42

    def __init__(self, api: Extractor):
        self.api = api
        self.nodes_map = api.data_store["nodes_map"]
        self.connections_map = api.data_store["connections_map"]
        self.demands = api.data_store["demands"]
        self.current_day = 0
        self.movements_plan = (
            {}
        )  # Movements scheduled (key: day, value: list of movements)
        self.in_transit_movements = []  # Movements that are in transit
        self.pending_demands = []  # Demands not yet fulfilled
        self.kpis = {"cost": 0, "co2": 0}
        self.session_id = None

    def send_start_data(self, api_url_start, api_key):
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

    def send_end_data(self, api_url_end=END_URL, api_key=API_KEY):
        headers = {"API-KEY": f"{api_key}"}
        try:
            response = requests.post(f"{api_url_end}", headers=headers)
            response.raise_for_status()  # Raise an error for bad responses
            logging.info("Delivery data sent successfully.")

            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Error sending delivery data: {e}")
            logging.debug(f"API Key: {api_key}")
            logging.debug(f"Headers: {headers}\n")
            return None

    def send_play_status(
        self,
        session_id,
        day,
        movements,
        api_url=ROUND_URL,
        api_key=API_KEY,
    ):
        headers = {
            "API-KEY": str(api_key),  # Ensure the API key is a string
            "SESSION-ID": str(session_id),  # Ensure the session ID is a string
            "Content-Type": "application/json",
        }
        # Prepare the payload with an empty movement for the first request
        delivery_data = {"day": day, "movements": []}
        try:
            response = requests.post(api_url, json=delivery_data, headers=headers)
            response.raise_for_status()  # Raise an error for bad responses
            logging.info("Delivery status sent successfully.")

            if day == self.NUMBER_OF_DAYS - 1:
                # pp("TOTAL KPIS: ", response.json())
                total_cost = response.json()["totalKpis"]["cost"]
                total_co2 = response.json()["totalKpis"]["co2"]
                print(f"--------- TOTAL COST: {total_cost} ---------")
                print(f"--------- TOTAL CO2:  {total_co2} ---------")
            # Process the response
            response_data = (
                response.json()
                if response.headers.get("Content-Type") == "application/json"
                else response.text.strip()
            )

            if response.status_code == 200:
                day_response = response.json()
                # Update KPIs and handle penalties if needed
                self.kpis["cost"] += day_response["deltaKpis"]["cost"]
                self.kpis["co2"] += day_response["deltaKpis"]["co2"]
                # Update demands
                new_demands = day_response.get("demand", [])
                for demand_info in new_demands:
                    demand = Demand(
                        id=str(uuid.uuid4()),
                        customer_id=demand_info["customerId"],
                        quantity=demand_info["amount"],
                        post_day=demand_info["postDay"],
                        start_delivery_day=demand_info["startDay"],
                        end_delivery_day=demand_info["endDay"],
                    )
                    self.pending_demands.append(demand)
                print(f"Day {self.current_day} played successfully.")
            else:
                print(
                    f"Failed to play round {self.current_day}: {response.status_code} {response.text}"
                )

            return response_data
        except requests.exceptions.RequestException as e:
            logging.error(f"Error sending delivery status: {e}")
            logging.debug(f"API Key: {api_key}\n")
            logging.debug(f"Headers: {headers}\n")
            return None

    def update_production(self):
        # Increase refinery stocks by their production rate
        for node in self.nodes_map.values():
            if isinstance(node, Refinery):
                node.stock += node.production
                if node.stock > node.capacity:
                    # Handle overflow penalty if necessary
                    pass

    def update_in_transit_movements(self):
        # Movements that arrive today
        arrived_movements = []
        for movement in self.in_transit_movements:
            if movement["arrival_day"] == self.current_day:
                # Update stocks at destination
                dest_node = self.nodes_map[movement["to_id"]]
                dest_node.stock += movement["amount"]
                arrived_movements.append(movement)
        # Remove arrived movements from in-transit list
        self.in_transit_movements = [
            m for m in self.in_transit_movements if m not in arrived_movements
        ]

    def update_internal_state(self, movements):
        # Update source stocks for movements
        for movement in movements:
            connection = self.connections_map[movement["connectionId"]]
            source_node = self.nodes_map[connection.from_id]
            source_node.stock -= movement["amount"]
            # We already updated stocks in plan_movements, so this might be redundant

    def find_possible_sources(self, customer):
        # Find storage tanks or refineries connected to the customer
        possible_sources = []
        for node in self.nodes_map.values():
            if isinstance(node, (StorageTank, Refinery)) and node.stock > 0:
                # Check if there's a connection to the customer
                for neighbour, connection in zip(node.neighbours, node.connections):
                    if neighbour.id == customer.id:
                        possible_sources.append((node, connection))
        # Sort by distance (assuming distance is a proxy for cost)
        possible_sources.sort(key=lambda x: x[1].distance)
        return possible_sources

    # INCOMPLETE
    """
    def plan_movements(self):
        # Simple heuristic: For each pending demand within delivery window, plan movements from closest storage tank
        movements = []

        # Filter demands that need to be fulfilled
        demands_to_fulfill = [
            d
            for d in self.pending_demands
            if d.remaining_quantity > 0
            and d.start_delivery_day <= self.current_day + 3 <= d.end_delivery_day
        ]

        for demand in demands_to_fulfill:
            customer = self.nodes_map[demand.customer_id]
            # Find the closest storage tank or refinery
            possible_sources = self.find_possible_sources(customer)
            if not possible_sources:
                continue  # No source available

            source_node, connection = possible_sources[
                0
            ]  # Choose the first one (closest)

            # Amount to move
            amount_to_move = min(
                demand.remaining_quantity, source_node.stock, customer.max_input
            )
            if amount_to_move <= 0:
                continue  # Can't move anything

            # Create movement
            movement = {"connectionId": connection.id, "amount": amount_to_move}
            movements.append(movement)

            # Update source stock
            source_node.stock -= amount_to_move

            # Schedule arrival
            arrival_day = self.current_day + connection.lead_time_days
            self.in_transit_movements.append(
                {
                    "from_id": source_node.id,
                    "to_id": customer.id,
                    "amount": amount_to_move,
                    "arrival_day": arrival_day,
                }
            )

            # Update demand
            demand.remaining_quantity -= amount_to_move

        return movements
    """

    def plan_movements(self):
        # Initialize variables to track movements and costs
        movements = []

        # Define initial weights for multi-criteria decision making
        cost_weight = 0.5
        emissions_weight = 0.3
        capacity_weight = 0.2

        # Sort customers by a combination of delivery end day and penalty
        self.pending_demands.sort(key=lambda d: (d.end_delivery_day, d.penalty))

        # Initial solution using heuristic
        for demand in self.pending_demands:
            customer = self.nodes_map[demand.customer_id]
            best_source = None
            best_score = float("inf")

            # Find the best source (refinery or storage tank) to fulfill the demand
            for source in self.nodes_map.values():
                if (
                    isinstance(source, (Refinery, StorageTank))
                    and source.stock >= demand.remaining_quantity
                ):
                    for connection in source.connections:
                        if connection.to_id == customer.id:
                            cost = demand.remaining_quantity * connection.cost
                            emissions = demand.remaining_quantity * connection.emissions
                            lead_time = connection.lead_time_days

                            # Check if delivery can be made within the window
                            if lead_time <= (
                                demand.end_delivery_day - self.current_day
                            ):
                                # Calculate a score based on multiple criteria
                                score = (
                                    cost * cost_weight
                                    + emissions * emissions_weight
                                    + (source.capacity - demand.remaining_quantity)
                                    * capacity_weight
                                )

                                # Check if this is the best option so far
                                if score < best_score:
                                    best_score = score
                                    best_source = (source, connection)

            if best_source:
                source_node, connection = best_source
                amount_to_move = min(
                    demand.remaining_quantity, source_node.stock, customer.max_input
                )

                # Create movement
                movement = {"connectionId": connection.id, "amount": amount_to_move}
                movements.append(movement)

                # Update source stock
                source_node.stock -= amount_to_move

                # Schedule arrival
                arrival_day = self.current_day + connection.lead_time_days
                self.in_transit_movements.append(
                    {
                        "from_id": source_node.id,
                        "to_id": customer.id,
                        "amount": amount_to_move,
                        "arrival_day": arrival_day,
                    }
                )

                # Update demand
                demand.remaining_quantity -= amount_to_move

        return movements

    """
    def optimize(self):
        # Main loop for each day
        for day in range(self.NUMBER_OF_DAYS):
            self.current_day = day
            print(f"\n--- Day {self.current_day} ---")

            # Update stocks (production at refineries)
            self.update_production()

            # Update in-transit movements
            self.update_in_transit_movements()

            # Plan movements for current day
            movements = self.plan_movements()

            # Submit movements to backend
            self.send_play_status(self.session_id, self.current_day, movements)

            # Update internal state for next day
            self.update_internal_state(movements)
    """

    def optimize(self):
        # Main loop for each day
        for day in range(self.NUMBER_OF_DAYS):
            self.current_day = day
            print(f"\n--- Day {self.current_day} ---")

            # Update stocks (production at refineries)
            self.update_production()

            # Update in-transit movements
            self.update_in_transit_movements()

            # Plan movements for current day
            movements = self.plan_movements()

            # Simulated Annealing
            temperature = 100
            cooling_rate = 0.95

            while temperature > 1:
                # Generate a neighboring solution
                new_movements = movements[:]
                if new_movements:
                    i = random.randint(0, len(new_movements) - 1)
                    # Randomly adjust a movement
                    new_movements[i] = {
                        "connectionId": new_movements[i]["connectionId"],
                        "amount": max(
                            0, new_movements[i]["amount"] + random.randint(-5, 5)
                        ),
                    }

                    # Calculate new cost and emissions
                    new_total_cost = sum(
                        m["amount"] * self.connections_map[m["connectionId"]].cost
                        for m in new_movements
                    )
                    new_total_emissions = sum(
                        m["amount"] * self.connections_map[m["connectionId"]].emissions
                        for m in new_movements
                    )

                    # Calculate the change in cost and emissions
                    delta_cost = new_total_cost - self.kpis["cost"]
                    delta_emissions = new_total_emissions - self.kpis["co2"]

                    # Acceptance probability
                    if (
                        delta_cost < 0
                        or delta_emissions < 0
                        or random.random() < math.exp(-delta_cost / temperature)
                    ):
                        movements = new_movements
                        self.kpis["cost"] = new_total_cost
                        self.kpis["co2"] = new_total_emissions

                # Decrease the temperature
                temperature *= cooling_rate

            # Submit movements to backend
            self.send_play_status(self.session_id, self.current_day, movements)

            # Update internal state for next day
            self.update_internal_state(movements)

    # Main simulation loop
    def run_simulation(self):
        self.session_id = self.send_start_data(self.START_URL, self.API_KEY)

        try:
            self.optimize()
        except Exception as e:
            logging.error(f"Error in simulation: {e}")

        self.send_end_data(self.END_URL, self.API_KEY)


if __name__ == "__main__":
    simple_api = Extractor()
    simple_api.parse_csv_info()
    simple_api.process_everything()
    radu_api = SolutionAPI(simple_api)
    logging.basicConfig(level=logging.DEBUG)
    radu_api.run_simulation()
