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
import networkx as nx
import time
from flask import Flask, send_file, jsonify
from flask_cors import CORS
import os
import csv


##################
# Define Constants
##################
ROOT_PATH = "./backend/api/eval-platform/src/main/resources/liquibase/data"
CONNECTIONS_PATH = ROOT_PATH + "/connections.csv"
CUSTOMERS_PATH = ROOT_PATH + "/customers.csv"
DEMANDS_PATH = ROOT_PATH + "/demands.csv"
REFINERIES_PATH = ROOT_PATH + "/refineries.csv"
TANKS_PATH = ROOT_PATH + "/tanks.csv"
TEAMS_PATH = ROOT_PATH + "/teams.csv"

CSV_PATH = "./static"
CSV_FILENAME = "map_data.csv"

# Initialize Flask app
app = Flask(__name__)
CORS(app)


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
        self.x = random.randint(50, 750)
        self.y = 50

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
        self.x = random.randint(50, 750)
        self.y = 200

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
        self.x = random.randint(50, 750)
        self.y = 500

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
        refineries_path = root_path + "/refineries.csv"
        tanks_path = root_path + "/tanks.csv"

        self.data_store["connections_data"] = self.extract_csv(connections_path)
        self.data_store["customers_data"] = self.extract_csv(customers_path)
        self.data_store["refineries_data"] = self.extract_csv(refineries_path)
        self.data_store["tanks_data"] = self.extract_csv(tanks_path)

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

    def process_everything(self):
        self.data_store["nodes_map"] = {}
        self.data_store["demands"] = []

        self.process_refineries()
        self.process_tanks()
        self.process_customers()
        self.process_connections()

    def post_connection_info(self) -> str:
        json_string = json.dumps(self.data_store["json_list"], indent=4)
        return json_string

    # Generate map data CSV for visualization
    def generate_map_csv(self):
        os.makedirs(CSV_PATH, exist_ok=True)
        filename = os.path.join(CSV_PATH, CSV_FILENAME)
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["id", "type", "x", "y", "from_id", "to_id", "lead_time_days", "amount"]
            )
            # Write points (refineries, tanks, customers)
            for node in self.data_store["nodes_map"].values():
                node_type = (
                    "refinery"
                    if isinstance(node, Refinery)
                    else "tank" if isinstance(node, StorageTank) else "gas_station"
                )
                writer.writerow([node.id, node_type, node.x, node.y, "", "", "", ""])
            # Write movements (connections)
            for connection in self.data_store["connections_map"].values():
                writer.writerow(
                    [
                        "",
                        "",
                        "",
                        "",
                        connection.from_id,
                        connection.to_id,
                        connection.lead_time_days,
                        connection.max_capacity,
                    ]
                )
        return filename


@app.route("/generate-csv", methods=["GET"])
def generate_csv():
    """
    Endpoint to generate and serve the CSV file.
    """
    filename = simple_api.generate_map_csv()
    return send_file(filename, as_attachment=True)


frontend_movements_per_day = {}


class SolutionAPI:
    BASE_URL = "https://localhost:8080/api/v1"
    START_URL = BASE_URL + "/session/start"
    ROUND_URL = BASE_URL + "/play/round"
    END_URL = BASE_URL + "/session/end"
    API_KEY = "7bcd6334-bc2e-4cbf-b9d4-61cb9e868869"
    NUMBER_OF_DAYS = 42

    def __init__(self, api: Extractor):
        self.api = api
        self.nodes_map = api.data_store["nodes_map"]
        self.connections_map = api.data_store["connections_map"]
        self.current_day = 0
        self.frontend_movements_per_day = {}
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

    @app.route("/generate-frontend-movements/day=<int:day>", methods=["GET"])
    def generate_frontend_movements(day):
        """
        Endpoint to retrieve movements for a specific day.
        """
        # Check if data exists for the given day
        if day in frontend_movements_per_day:
            return jsonify(
                frontend_movements_per_day[day]
            )  # Return data as JSON response
        else:
            return (
                jsonify({"error": "Data not found for specified day"}),
                404,
            )  # Return 404 if not found

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
        delivery_data = {"day": day, "movements": movements}
        frontend_movements = [
            {
                "from_id": self.connections_map[movement["connectionId"]].from_id,
                "to_id": self.connections_map[movement["connectionId"]].to_id,
                "lead_time_days": self.connections_map[
                    movement["connectionId"]
                ].lead_time_days,
                "type": 
                    self.connections_map[movement["connectionId"]].connection_type
                ,
            }
            for movement in movements
        ]
        frontend_movements_per_day[day] = frontend_movements
        try:
            response = requests.post(api_url, json=delivery_data, headers=headers)
            response.raise_for_status()  # Raise an error for bad responses
            logging.info("Delivery status sent successfully.")

            if day == self.NUMBER_OF_DAYS - 1:
                print("TOTAL KPIS: ", response.json())
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

    def get_min_lead_time(self, customer):
        """
        Returns the minimum lead time from any source to the customer.
        """
        min_lead_time = float("inf")
        for node in self.nodes_map.values():
            for neighbour, connection in zip(node.neighbours, node.connections):
                if neighbour.id == customer.id:
                    if connection.lead_time_days < min_lead_time:
                        min_lead_time = connection.lead_time_days
        return min_lead_time if min_lead_time != float("inf") else 0

    def maximum_flow(self, G, source, sink):
        """
        Computes the maximum flow in the network G from source to sink.
        """

        # Use Edmonds-Karp algorithm for maximum flow
        max_flow_value, flow_dict = nx.maximum_flow(
            G, source, sink, flow_func=nx.algorithms.flow.edmonds_karp
        )
        return max_flow_value, flow_dict

    def flow_to_movements(
        self, flow_dict, node_daily_outputs, node_daily_inputs, connection_in_transit
    ):
        """
        Converts the flow dictionary into a list of movements, considering capacities and constraints.
        """
        movements = []

        # Iterate over the flow_dict to create movements
        for from_node_id, to_nodes in flow_dict.items():
            if from_node_id in ["source", "sink"]:
                continue
            for to_node_id, flow_amount in to_nodes.items():
                if flow_amount > 0 and to_node_id not in ["source", "sink"]:
                    connection_id = self.get_connection_id(from_node_id, to_node_id)
                    if connection_id is None:
                        continue  # Invalid connection

                    connection = self.connections_map[connection_id]
                    from_node = self.nodes_map[from_node_id]
                    to_node = self.nodes_map[to_node_id]

                    # Check capacities and constraints
                    available_output = getattr(
                        from_node, "max_output", float("inf")
                    ) - node_daily_outputs.get(from_node_id, 0)
                    available_input = getattr(
                        to_node, "max_input", float("inf")
                    ) - node_daily_inputs.get(to_node_id, 0)
                    to_node_available_capacity = (
                        getattr(to_node, "capacity", float("inf")) - to_node.stock
                    )
                    conn_available_capacity = (
                        connection.max_capacity
                        - connection_in_transit.get(connection_id, 0)
                    )

                    amount_to_move = min(
                        flow_amount,
                        available_output,
                        available_input,
                        to_node_available_capacity,
                        conn_available_capacity,
                    )

                    if amount_to_move <= 0:
                        continue  # Cannot move any amount without violating constraints

                    # Update tracking dictionaries
                    node_daily_outputs[from_node_id] = (
                        node_daily_outputs.get(from_node_id, 0) + amount_to_move
                    )
                    node_daily_inputs[to_node_id] = (
                        node_daily_inputs.get(to_node_id, 0) + amount_to_move
                    )
                    connection_in_transit[connection_id] = (
                        connection_in_transit.get(connection_id, 0) + amount_to_move
                    )

                    # Create movement
                    movement = {"connectionId": connection_id, "amount": amount_to_move}
                    movements.append(movement)

                    # Schedule arrival
                    arrival_day = self.current_day + connection.lead_time_days
                    self.in_transit_movements.append(
                        {
                            "from_id": from_node_id,
                            "to_id": to_node_id,
                            "amount": amount_to_move,
                            "arrival_day": arrival_day,
                        }
                    )

                    # Update stocks
                    from_node.stock -= amount_to_move

        return movements

    def build_flow_network(
        self, connection_in_transit, node_daily_outputs, node_daily_inputs
    ):
        """
        Builds a flow network for the maximum flow algorithm, incorporating storage tanks.
        """
        import networkx as nx

        G = nx.DiGraph()

        # Add nodes
        G.add_node("source")
        G.add_node("sink")
        for node in self.nodes_map.values():
            G.add_node(node.id)

        # Add edges from source to refineries
        for refinery in [n for n in self.nodes_map.values() if isinstance(n, Refinery)]:
            # Update refinery stock with daily production
            refinery.stock += refinery.production

            # Calculate available output capacity
            available_output = refinery.max_output - node_daily_outputs.get(
                refinery.id, 0
            )
            # Capacity is the minimum of available output and available stock
            available_stock = refinery.stock
            capacity = min(available_output, available_stock)
            if capacity > 0:
                G.add_edge("source", refinery.id, capacity=capacity)

        # Add edges between nodes (refineries, storage tanks, customers)
        for node in self.nodes_map.values():
            for neighbour, connection in zip(node.neighbours, node.connections):
                # Calculate available capacities
                conn_available_capacity = (
                    connection.max_capacity
                    - connection_in_transit.get(connection.id, 0)
                )
                if conn_available_capacity <= 0:
                    continue  # No capacity left on connection

                node_available_output = getattr(
                    node, "max_output", float("inf")
                ) - node_daily_outputs.get(node.id, 0)
                neighbour_available_input = getattr(
                    neighbour, "max_input", float("inf")
                ) - node_daily_inputs.get(neighbour.id, 0)
                neighbour_available_capacity = (
                    getattr(neighbour, "capacity", float("inf")) - neighbour.stock
                )

                capacity = min(
                    conn_available_capacity,
                    node_available_output,
                    neighbour_available_input,
                    neighbour_available_capacity,
                )
                if capacity > 0:
                    G.add_edge(node.id, neighbour.id, capacity=capacity)

        # Add edges from customers to sink
        for customer in [n for n in self.nodes_map.values() if isinstance(n, Customer)]:
            # Demand is sum of remaining quantities for the customer within delivery window
            total_demand = sum(
                d.remaining_quantity
                for d in self.pending_demands
                if d.customer_id == customer.id
                and d.start_delivery_day
                <= self.current_day + self.get_min_lead_time(customer)
                <= d.end_delivery_day
            )
            if total_demand > 0:
                G.add_edge(customer.id, "sink", capacity=total_demand)

        return G

    def get_connection_id(self, from_id, to_id):
        """
        Retrieves the connection ID between two nodes.
        """
        for conn_id, connection in self.connections_map.items():
            if connection.from_id == from_id and connection.to_id == to_id:
                return conn_id
        return None

    def plan_movements(self):
        """
        Plan movements for the current day, ensuring that:
        - No penalties are incurred at the end of the day.
        - Movements are scheduled to prevent refinery overflows.
        - Movements are scheduled to fulfill customer demands.
        - Movements are only scheduled if they can be executed without causing penalties.
        """
        movements = []

        # Initialize tracking dictionaries
        node_daily_outputs = (
            {}
        )  # Key: node ID, Value: total output scheduled for the day
        node_daily_inputs = {}  # Key: node ID, Value: total input scheduled for the day
        connection_in_transit = {}  # Key: connection ID, Value: total amount in transit

        # Update with existing in-transit movements
        for movement in self.in_transit_movements:
            from_id = movement["from_id"]
            node_daily_outputs[from_id] = (
                node_daily_outputs.get(from_id, 0) + movement["amount"]
            )

            to_id = movement["to_id"]
            node_daily_inputs[to_id] = (
                node_daily_inputs.get(to_id, 0) + movement["amount"]
            )

            connection_id = self.get_connection_id(from_id, to_id)
            if connection_id:
                connection_in_transit[connection_id] = (
                    connection_in_transit.get(connection_id, 0) + movement["amount"]
                )

        # Process refineries to prevent overflows
        for node in self.nodes_map.values():
            if isinstance(node, Refinery):
                # Increase stock by production
                node.stock += node.production

                # Check for potential overflow
                if node.stock > node.capacity:
                    overflow_amount = node.stock - node.capacity

                    # Attempt to move overflow to connected storage tanks
                    possible_tanks = [
                        (neighbour, connection)
                        for neighbour, connection in zip(
                            node.neighbours, node.connections
                        )
                        if isinstance(neighbour, StorageTank)
                    ]

                    for tank, connection in possible_tanks:
                        # Calculate available capacities
                        tank_available_capacity = tank.capacity - tank.stock
                        connection_available_capacity = (
                            connection.max_capacity
                            - connection_in_transit.get(connection.id, 0)
                        )
                        node_available_output = (
                            node.max_output - node_daily_outputs.get(node.id, 0)
                        )
                        tank_available_input = tank.max_input - node_daily_inputs.get(
                            tank.id, 0
                        )

                        # Determine the maximum amount we can move without causing penalties
                        amount_to_move = min(
                            overflow_amount,
                            tank_available_capacity,
                            connection_available_capacity,
                            node_available_output,
                            tank_available_input,
                        )

                        if amount_to_move <= 0:
                            continue  # Cannot move any amount without causing penalties

                        # Create movement
                        movement = {
                            "connectionId": connection.id,
                            "amount": amount_to_move,
                        }
                        movements.append(movement)

                        # Update tracking dictionaries
                        node_daily_outputs[node.id] = (
                            node_daily_outputs.get(node.id, 0) + amount_to_move
                        )
                        node_daily_inputs[tank.id] = (
                            node_daily_inputs.get(tank.id, 0) + amount_to_move
                        )
                        connection_in_transit[connection.id] = (
                            connection_in_transit.get(connection.id, 0) + amount_to_move
                        )

                        # Schedule arrival
                        arrival_day = self.current_day + connection.lead_time_days
                        self.in_transit_movements.append(
                            {
                                "from_id": node.id,
                                "to_id": tank.id,
                                "amount": amount_to_move,
                                "arrival_day": arrival_day,
                            }
                        )

                        # Update stocks
                        node.stock -= amount_to_move
                        overflow_amount -= amount_to_move

                        if overflow_amount <= 0:
                            break  # Overflow has been handled

                # Ensure refinery stock does not exceed capacity
                if node.stock > node.capacity:
                    # Even after attempting to move excess, overflow remains
                    # Set stock to capacity to prevent further penalties
                    node.stock = node.capacity

        # Now, plan movements to fulfill customer demands
        # First, get demands that need to be fulfilled
        demands_to_fulfill = []
        for demand in self.pending_demands:
            # Calculate the earliest possible arrival day considering lead times
            customer = self.nodes_map[demand.customer_id]
            min_lead_time = self.get_min_lead_time(customer)
            earliest_arrival_day = self.current_day + min_lead_time

            # Check if the demand can be delivered within its delivery window
            if (
                demand.start_delivery_day
                <= earliest_arrival_day
                <= demand.end_delivery_day
            ):
                demands_to_fulfill.append(demand)

        # For each demand, plan movements from sources to customer
        for demand in demands_to_fulfill:
            customer = self.nodes_map[demand.customer_id]
            remaining_quantity = demand.remaining_quantity

            # Find possible sources
            possible_sources = []
            for node in self.nodes_map.values():
                if isinstance(node, (Refinery, StorageTank)) and node.stock > 0:
                    # Check if there's a connection to the customer
                    for neighbour, connection in zip(node.neighbours, node.connections):
                        if neighbour.id == customer.id:
                            possible_sources.append((node, connection))

            # Sort possible sources by lead time (shortest first)
            possible_sources.sort(key=lambda x: x[1].lead_time_days)

            for source_node, connection in possible_sources:
                if remaining_quantity <= 0:
                    break

                # Calculate available capacities
                connection_available_capacity = (
                    connection.max_capacity
                    - connection_in_transit.get(connection.id, 0)
                )
                source_available_output = getattr(
                    source_node, "max_output", float("inf")
                ) - node_daily_outputs.get(source_node.id, 0)
                customer_available_input = customer.max_input - node_daily_inputs.get(
                    customer.id, 0
                )
                source_stock = source_node.stock

                amount_to_move = min(
                    remaining_quantity,
                    connection_available_capacity,
                    source_available_output,
                    customer_available_input,
                    source_stock,
                )

                if amount_to_move <= 0:
                    continue  # Cannot move any amount

                # Create movement
                movement = {"connectionId": connection.id, "amount": amount_to_move}
                movements.append(movement)

                # Update tracking dictionaries
                node_daily_outputs[source_node.id] = (
                    node_daily_outputs.get(source_node.id, 0) + amount_to_move
                )
                node_daily_inputs[customer.id] = (
                    node_daily_inputs.get(customer.id, 0) + amount_to_move
                )
                connection_in_transit[connection.id] = (
                    connection_in_transit.get(connection.id, 0) + amount_to_move
                )

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

                # Update stocks
                source_node.stock -= amount_to_move
                remaining_quantity -= amount_to_move
                demand.remaining_quantity -= amount_to_move

        # Remove demands that have been fully fulfilled
        self.pending_demands = [
            d for d in self.pending_demands if d.remaining_quantity > 0
        ]

        return movements

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

            # time.sleep(5)

    # Main simulation loop
    def run_simulation(self):
        self.session_id = self.send_start_data(self.START_URL, self.API_KEY)

        try:
            self.optimize()
        except Exception as e:
            logging.error(f"Error in simulation: {e}")

        self.send_end_data(self.END_URL, self.API_KEY)


if __name__ == "__main__":
    # Check if the script is running with the Flask reloader
    import os

    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        # This block will only run once, when the reloader is not active
        simple_api = Extractor()
        simple_api.parse_csv_info()
        simple_api.process_everything()
        radu_api = SolutionAPI(simple_api)
        logging.basicConfig(level=logging.DEBUG)

        # Run the simulation only once
        radu_api.run_simulation()

    # Start the Flask app
    app.run(port=5000, debug=True)
