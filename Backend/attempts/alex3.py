import heapq
from collections import defaultdict
import logging
import requests
import json
import os
import csv
from flask import Flask
from flask_cors import CORS
import pandas as pd
from enum import Enum

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
    REFINERY = 1
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
        connection_type,
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
    def __init__(self, neighbours, connections):
        self.neighbours = neighbours
        self.connections = connections

    def add_neighbour(self, neighbour, connection):
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
        node_type,
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
        node_type,
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


class Customer(Node):
    def __init__(
        self,
        id,
        name,
        max_input,
        over_input_penalty,
        late_delivery_penalty,
        early_delivery_penalty,
        node_type,
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


class Demand:
    def __init__(self, demand_id, target, capacity, start_day, end_day):
        self.demand_id = demand_id
        self.target = target
        self.capacity = capacity
        self.start_day = start_day
        self.end_day = end_day


class Extractor:
    def __init__(self):
        self.data_store = {
            "nodes_map": {},
            "connections_map": {},
            "demands": [],
            "edge_list": [],
            "json_list": [],
        }

    def process_refineries(self):
        refineries_data = pd.read_csv(REFINERIES_PATH)
        for _, row in refineries_data.iterrows():
            node_type = (
                NodeType.REFINERY
                if row.node_type == "REFINERY"
                else NodeType.STORAGE_TANK
            )
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
                node_type,
            )
            self.data_store["nodes_map"][row.id] = new_refinery

    def process_tanks(self):
        tanks_data = pd.read_csv(TANKS_PATH)
        for _, row in tanks_data.iterrows():
            node_type = (
                NodeType.STORAGE_TANK
                if row.node_type == "STORAGE_TANK"
                else NodeType.REFINERY
            )
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
                node_type,
            )
            self.data_store["nodes_map"][row.id] = new_tank

    def process_customers(self):
        customers_data = pd.read_csv(CUSTOMERS_PATH)
        for _, row in customers_data.iterrows():
            node_type = (
                NodeType.CUSTOMER if row.node_type == "CUSTOMER" else NodeType.REFINERY
            )
            new_customer = Customer(
                row.id,
                row.name,
                row.max_input,
                row.over_input_penalty,
                row.late_delivery_penalty,
                row.early_delivery_penalty,
                node_type,
            )
            self.data_store["nodes_map"][row.id] = new_customer

    def process_connections(self):
        connections_data = pd.read_csv(CONNECTIONS_PATH)
        for _, row in connections_data.iterrows():
            new_connection = Connection(
                row.id,
                row.from_id,
                row.to_id,
                row.distance,
                row.lead_time_days,
                ConnectionType[row.connection_type],
                row.max_capacity,
            )
            self.data_store["connections_map"][row.id] = new_connection
            self.data_store["nodes_map"][row.from_id].add_neighbour(
                self.data_store["nodes_map"][row.to_id], new_connection
            )
            self.data_store["edge_list"].append((row.from_id, row.to_id))
            self.data_store["json_list"].append(
                {
                    "from_id": row.from_id,
                    "from_node_type": self.data_store["nodes_map"][
                        row.from_id
                    ].node_type.name,
                    "to_id": row.to_id,
                    "to_node_type": self.data_store["nodes_map"][
                        row.to_id
                    ].node_type.name,
                }
            )

    def process_demands(self):
        demands_data = pd.read_csv(DEMANDS_PATH)
        for _, row in demands_data.iterrows():
            new_demand = Demand(
                row.id,
                row.customer_id,
                row.quantity,
                row.start_delivery_day,
                row.end_delivery_day,
            )
            self.data_store["demands"].append(new_demand)

    def process_everything(self):
        self.process_refineries()
        self.process_tanks()
        self.process_customers()
        self.process_connections()
        self.process_demands()

    def generate_map_csv(self):
        os.makedirs(CSV_PATH, exist_ok=True)
        filename = os.path.join(CSV_PATH, CSV_FILENAME)
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["id", "type", "x", "y", "from_id", "to_id", "lead_time_days", "amount"]
            )
            for node in self.data_store["nodes_map"].values():
                node_type = (
                    "refinery"
                    if isinstance(node, Refinery)
                    else "tank" if isinstance(node, StorageTank) else "gas_station"
                )
                writer.writerow([node.id, node_type, node.x, node.y, "", "", "", ""])
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
        self.frontend_movements_per_day = {}
        self.movements_plan = {}
        self.in_transit_movements = []
        self.pending_demands = []
        self.kpis = {"cost": 0, "co2": 0}
        self.session_id = None

    def send_start_data(self, api_url_start, api_key):
        headers = {"API-KEY": f"{api_key}"}
        try:
            response = requests.post(api_url_start, headers=headers)
            response.raise_for_status()
            logging.info("Session started successfully.")
            session_id = response.text.strip()
            if session_id:
                logging.info(f"Session ID received: {session_id}")
            else:
                logging.warning("Session ID not found in the response.")
            return session_id
        except requests.exceptions.RequestException as e:
            logging.error(f"Error starting session: {e}")
            return None

    def send_end_data(self, api_url_end=END_URL, api_key=API_KEY):
        headers = {"API-KEY": f"{api_key}"}
        try:
            response = requests.post(api_url_end, headers=headers)
            response.raise_for_status()
            logging.info("Session ended successfully.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Error ending session: {e}")

    def min_cost_maximal_matching(self):
        adj_list = defaultdict(list)
        for connection in self.connections_map.values():
            adj_list[connection.from_id].append(connection)

        demand_received = {d.demand_id: 0 for d in self.demands}
        daily_intake = {
            t.id: defaultdict(int)
            for t in self.nodes_map.values()
            if isinstance(t, Customer)
        }

        matching = []
        penalty_costs = 0
        pq = []

        for source in self.nodes_map.values():
            if isinstance(source, (Refinery, StorageTank)):
                for connection in adj_list[source.id]:
                    for demand in self.demands:
                        if demand.target == connection.to_id:
                            if (
                                demand.start_day
                                <= connection.lead_time_days
                                <= demand.end_day
                            ):
                                remaining_demand = (
                                    demand.capacity - demand_received[demand.demand_id]
                                )
                                flow = min(
                                    connection.max_capacity,
                                    remaining_demand,
                                    source.max_output,
                                    source.stock,
                                )

                                if flow > 0:
                                    heapq.heappush(
                                        pq,
                                        (
                                            connection.distance,
                                            connection.lead_time_days,
                                            flow,
                                            [connection],
                                            demand,
                                        ),
                                    )

        while pq:
            current_cost, current_delay, current_flow, path, demand = heapq.heappop(pq)
            last_connection = path[-1]
            target_id = last_connection.to_id
            target_node = self.nodes_map[target_id]

            day = current_delay
            allocated_flow = 0

            while current_flow > 0 and day <= demand.end_day:
                daily_capacity_left = (
                    target_node.max_input - daily_intake[target_id][day]
                )
                flow_for_day = min(
                    current_flow,
                    daily_capacity_left,
                    demand.capacity - demand_received[demand.demand_id],
                )

                if flow_for_day > 0:
                    daily_intake[target_id][day] += flow_for_day
                    demand_received[demand.demand_id] += flow_for_day
                    allocated_flow += flow_for_day
                    current_flow -= flow_for_day

                    matching.append(
                        (
                            [(c.from_id, c.to_id) for c in path],
                            flow_for_day,
                            current_cost,
                            demand.demand_id,
                        )
                    )

                day += 1

            if demand_received[demand.demand_id] >= demand.capacity:
                if day - 1 < demand.start_day:
                    penalty_costs += target_node.early_delivery_penalty
                elif day - 1 > demand.end_day:
                    penalty_costs += target_node.late_delivery_penalty
            elif day > demand.end_day:
                penalty_costs += target_node.late_delivery_penalty

            for d in range(demand.start_day, demand.end_day + 1):
                if daily_intake[target_id][d] > target_node.max_input:
                    penalty_costs += target_node.over_input_penalty * (
                        daily_intake[target_id][d] - target_node.max_input
                    )

            if last_connection.to_id in [
                source.id
                for source in self.nodes_map.values()
                if isinstance(source, (Refinery, StorageTank))
            ]:
                source_node = self.nodes_map[last_connection.to_id]
                for connection in adj_list[source_node.id]:
                    next_target = connection.to_id
                    new_delay = current_delay + connection.lead_time_days
                    new_cost = current_cost + connection.distance

                    if new_delay < demand.start_day or new_delay > demand.end_day:
                        continue

                    remaining_demand = (
                        demand.capacity - demand_received[demand.demand_id]
                    )
                    new_flow = min(
                        current_flow,
                        connection.max_capacity,
                        remaining_demand,
                        source_node.stock,
                        source_node.max_output,
                    )

                    if new_flow > 0:
                        heapq.heappush(
                            pq,
                            (
                                new_cost,
                                new_delay,
                                new_flow,
                                path + [connection],
                                demand,
                            ),
                        )

                    source_node.stock -= allocated_flow
                    if source_node.stock < 0:
                        penalty_costs += source_node.underflow_penalty
                    elif source_node.stock > source_node.capacity:
                        penalty_costs += source_node.overflow_penalty

        total_cost = (
            sum(flow * path_cost for _, flow, path_cost, _ in matching) + penalty_costs
        )
        return matching, total_cost

    def optimize(self):
        for day in range(self.NUMBER_OF_DAYS):
            self.current_day = day
            print(f"\n--- Day {self.current_day} ---")

            self.update_production()
            self.update_in_transit_movements()

            movements, total_cost = self.min_cost_maximal_matching()
            self.kpis["cost"] += total_cost

            self.send_play_status(self.session_id, self.current_day, movements)
            self.update_internal_state(movements)

    def run_simulation(self):
        self.session_id = self.send_start_data(self.START_URL, self.API_KEY)

        try:
            self.optimize()
        except Exception as e:
            logging.error(f"Error in simulation: {e}")

        self.send_end_data(self.END_URL, self.API_KEY)


if __name__ == "__main__":
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        simple_api = Extractor()
        simple_api.process_everything()
        radu_api = SolutionAPI(simple_api)
        logging.basicConfig(level=logging.DEBUG)

        radu_api.run_simulation()

    app.run(port=5000, debug=True)
