import heapq
from collections import defaultdict
import logging
import requests
import uuid
import random
import json
import time
from flask import Flask
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)


################
# Define Classes
################
class Edge:
    def __init__(self, source, target, cost, delay, capacity):
        self.source = source
        self.target = target
        self.cost = cost
        self.delay = delay
        self.capacity = capacity


class Demand:
    def __init__(self, demand_id, target, capacity, start_day, end_day):
        self.demand_id = demand_id
        self.target = target
        self.capacity = capacity
        self.start_day = start_day
        self.end_day = end_day


class TargetNode:
    def __init__(
        self,
        id,
        max_input,
        over_input_penalty,
        late_delivery_penalty,
        early_delivery_penalty,
        node_type,
    ):
        self.id = id
        self.max_input = max_input
        self.over_input_penalty = over_input_penalty
        self.late_delivery_penalty = late_delivery_penalty
        self.early_delivery_penalty = early_delivery_penalty
        self.node_type = node_type


class SourceNode:
    def __init__(
        self,
        id,
        capacity,
        max_input,
        max_output,
        overflow_penalty,
        underflow_penalty,
        over_input_penalty,
        over_output_penalty,
        initial_stock,
    ):
        self.id = id
        self.capacity = capacity
        self.max_input = max_input
        self.max_output = max_output
        self.overflow_penalty = overflow_penalty
        self.underflow_penalty = underflow_penalty
        self.over_input_penalty = over_input_penalty
        self.over_output_penalty = over_output_penalty
        self.stock = initial_stock


def min_cost_maximal_matching(sources, targets, edges, demands):
    adj_list = defaultdict(list)
    for edge in edges:
        adj_list[edge.source].append(edge)

    # Track the total received resources for each demand and daily intake limits
    demand_received = {d.demand_id: 0 for d in demands}
    daily_intake = {
        t.id: defaultdict(int) for t in targets
    }  # {target: {day: intake_amount}}

    # Result set of edges that form the matching and penalty tracker
    matching = []
    penalty_costs = 0

    # Priority queue for BFS-like exploration with (cost, delay, flow, path, demand) entries
    pq = []

    # Initialize paths from source nodes
    for source in sources:
        for edge in adj_list[source.id]:
            for demand in demands:
                if demand.target == edge.target:
                    # Only consider paths that fall within the demand's allowable delivery window
                    if demand.start_day <= edge.delay <= demand.end_day:
                        remaining_demand = (
                            demand.capacity - demand_received[demand.demand_id]
                        )
                        flow = min(
                            edge.capacity,
                            remaining_demand,
                            source.max_output,
                            source.stock,
                        )

                        if flow > 0:
                            heapq.heappush(
                                pq, (edge.cost, edge.delay, flow, [edge], demand)
                            )

    # Process paths while respecting constraints
    while pq:
        current_cost, current_delay, current_flow, path, demand = heapq.heappop(pq)

        # Get the last edge in the current path
        last_edge = path[-1]
        target_id = last_edge.target
        target_node = next(t for t in targets if t.id == target_id)

        # Attempt to distribute the flow over multiple days within the demand's time window
        day = current_delay  # Start at the delivery day
        allocated_flow = 0

        while current_flow > 0 and day <= demand.end_day:
            # Calculate the remaining capacity for this day, respecting max_input_per_day
            daily_capacity_left = target_node.max_input - daily_intake[target_id][day]
            flow_for_day = min(
                current_flow,
                daily_capacity_left,
                demand.capacity - demand_received[demand.demand_id],
            )

            if flow_for_day > 0:
                # Update allocations
                daily_intake[target_id][day] += flow_for_day
                demand_received[demand.demand_id] += flow_for_day
                allocated_flow += flow_for_day
                current_flow -= flow_for_day

                # Add this path to the matching if it contributes flow
                matching.append(
                    (
                        [(e.source, e.target) for e in path],
                        flow_for_day,
                        current_cost,
                        demand.demand_id,
                    )
                )

            # Move to the next day if more flow remains to be allocated
            day += 1

        # Calculate penalties for this specific demand based on target-specific penalties
        if demand_received[demand.demand_id] >= demand.capacity:
            if day - 1 < demand.start_day:
                penalty_costs += target_node.early_delivery_penalty  # Early delivery
            elif day - 1 > demand.end_day:
                penalty_costs += target_node.late_delivery_penalty  # Late delivery
        elif day > demand.end_day:
            penalty_costs += (
                target_node.late_delivery_penalty
            )  # Demand not fully met by deadline

        # Track any penalties for exceeding max input per day
        for d in range(demand.start_day, demand.end_day + 1):
            if daily_intake[target_id][d] > target_node.max_input:
                penalty_costs += target_node.over_input_penalty * (
                    daily_intake[target_id][d] - target_node.max_input
                )

        # Expand paths from the current node if it's a source, tracking stock levels
        if last_edge.target in [source.id for source in sources]:
            source_node = next(s for s in sources if s.id == last_edge.target)
            for edge in adj_list[source_node.id]:
                next_target = edge.target
                new_delay = current_delay + edge.delay
                new_cost = current_cost + edge.cost

                # Only consider paths within the demand's allowable delivery window
                if new_delay < demand.start_day or new_delay > demand.end_day:
                    continue

                # Calculate available flow based on the next edge, remaining demand, and source stock
                remaining_demand = demand.capacity - demand_received[demand.demand_id]
                new_flow = min(
                    current_flow,
                    edge.capacity,
                    remaining_demand,
                    source_node.stock,
                    source_node.max_output,
                )

                if new_flow > 0:
                    heapq.heappush(
                        pq, (new_cost, new_delay, new_flow, path + [edge], demand)
                    )

                # Update source stock and track penalties if stock goes out of bounds
                source_node.stock -= allocated_flow
                if source_node.stock < 0:
                    penalty_costs += source_node.underflow_penalty
                elif source_node.stock > source_node.capacity:
                    penalty_costs += source_node.overflow_penalty

    # Calculate the total cost of the matching
    total_cost = (
        sum(flow * path_cost for _, flow, path_cost, _ in matching) + penalty_costs
    )

    return matching, total_cost


# Example usage
sources = [
    SourceNode(
        id="S1",
        capacity=100,
        max_input=40,
        max_output=30,
        overflow_penalty=10,
        underflow_penalty=15,
        over_input_penalty=5,
        over_output_penalty=7,
        initial_stock=50,
    ),
    SourceNode(
        id="S2",
        capacity=80,
        max_input=30,
        max_output=25,
        overflow_penalty=12,
        underflow_penalty=10,
        over_input_penalty=4,
        over_output_penalty=6,
        initial_stock=30,
    ),
]
targets = [
    TargetNode(
        id="T1",
        max_input=30,
        over_input_penalty=5,
        late_delivery_penalty=20,
        early_delivery_penalty=15,
        node_type="A",
    ),
    TargetNode(
        id="T2",
        max_input=25,
        over_input_penalty=6,
        late_delivery_penalty=25,
        early_delivery_penalty=10,
        node_type="B",
    ),
]
edges = [
    Edge("S1", "T1", cost=3, delay=2, capacity=50),
    Edge("S1", "T2", cost=1, delay=1, capacity=30),
    Edge("S2", "T1", cost=2, delay=2, capacity=60),
    Edge("S2", "T2", cost=2, delay=3, capacity=40),
]
demands = [
    Demand(demand_id=1, target="T1", capacity=80, start_day=2, end_day=4),
    Demand(demand_id=2, target="T2", capacity=50, start_day=1, end_day=3),
]

matching, total_cost = min_cost_maximal_matching(sources, targets, edges, demands)

print("Matching result (path, daily flow allocation, cost, demand_id):")
for match in matching:
    print(match)
print("Total cost (including penalties):", total_cost)

if __name__ == "__main__":
    # Start the Flask app
    app.run(port=5000, debug=True)
