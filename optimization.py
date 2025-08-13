# optimization.py
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from typing import List, Optional
from config import Config
from exceptions import OptimizationError
from custom_logging import logger
import random
import numpy as np

class VRPOptimizer:
    def __init__(self):
        self.seed = Config.get("OPTIMIZATION_SEED", 12345)
        random.seed(self.seed)
        np.random.seed(self.seed)
    
    def solve_vrp(
        self,
        distance_matrix: List[List[float]],
        time_matrix: List[List[int]],
        max_time: int = 1000,
        num_vehicles: int = 1,
        depot: int = 0
    ) -> Optional[List[List[int]]]:
        """Solve Vehicle Routing Problem with time constraints"""
        size = len(distance_matrix)
        if size <= 1:
            logger.warning("Not enough locations for VRP")
            return None

        # Create routing model
        manager = pywrapcp.RoutingIndexManager(size, num_vehicles, depot)
        routing = pywrapcp.RoutingModel(manager)

        # Distance callback
        def distance_callback(from_index: int, to_index: int) -> int:
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(distance_matrix[from_node][to_node] * 1000)

        transit_cb_idx = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)

        # Time callback
        def time_callback(from_index: int, to_index: int) -> int:
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return time_matrix[from_node][to_node]

        time_cb_idx = routing.RegisterTransitCallback(time_callback)
        routing.AddDimension(time_cb_idx, 0, max_time, True, "Time")

        # Configure search parameters
        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        params.time_limit.seconds = 10
        params.log_search = Config.DEBUG

        # Solve the problem
        solution = routing.SolveWithParameters(params)
        if not solution:
            logger.warning("No solution found for VRP")
            return None

        # Extract routes
        routes = []
        for vehicle_id in range(num_vehicles):
            index = routing.Start(vehicle_id)
            route = []
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                route.append(node)
                index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
            routes.append(route)

        logger.info("VRP solved successfully", num_routes=len(routes))
        return routes if num_vehicles > 1 else routes[0]