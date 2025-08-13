# routing.py
import math
import requests
from typing import Any, Dict, List, Optional, Tuple
from config import Config
from exceptions import OSRMError
from custom_logging import logger
import redis
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

class RouteCalculator:
    def __init__(self):
        self.cache = redis.Redis(host='localhost', port=6379, db=0)
    
    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great-circle distance between two points"""
        R = 6371  # Earth radius in km
        phi1, phi2 = map(math.radians, (lat1, lat2))
        d_phi = math.radians(lat2 - lat1)
        d_lambda = math.radians(lon2 - lon1)

        a = math.sin(d_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(d_lambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c
    
    def get_osrm_url(self, country: str) -> str:
        """Get OSRM server URL for the given country"""
        url = Config.OSRM_SERVERS.get(country)
        if not url:
            logger.warning("No OSRM server configured for country", country=country)
            url = Config.OSRM_SERVERS["Belgique"]  # Fallback
        return url
    
    def get_cached_route(self, key: str) -> Optional[Tuple[float, float]]:
        """Get cached route distance and duration"""
        try:
            cached = self.cache.get(key)
            return json.loads(cached) if cached else None
        except redis.RedisError as e:
            logger.warning("Cache access failed", error=str(e))
            return None
    
    def set_cached_route(self, key: str, value: Tuple[float, float]) -> None:
        """Cache route distance and duration"""
        try:
            self.cache.setex(key, Config.CACHE_TTL, json.dumps(value))
        except redis.RedisError as e:
            logger.warning("Cache set failed", error=str(e))
    
    def get_osrm_route_distance_duration(
        self, 
        lat1: float, 
        lon1: float, 
        lat2: float, 
        lon2: float, 
        country: str
    ) -> Tuple[float, float]:
        """Get route distance (km) and duration (min) from OSRM"""
        key = f"route_{lat1}_{lon1}_{lat2}_{lon2}_{country}"
        cached = self.get_cached_route(key)
        if cached:
            return cached
        
        base_url = self.get_osrm_url(country)
        query_url = f"{base_url}/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false"

        try:
            resp = requests.get(query_url, timeout=Config.OSRM_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            
            if not data.get("routes"):
                raise OSRMError("No routes in response")
                
            route = data["routes"][0]
            distance_km = route["distance"] / 1000
            duration_min = route["duration"] / 60
            
            if Config.DEBUG:
                logger.debug(
                    "OSRM route calculated",
                    country=country,
                    distance_km=distance_km,
                    duration_min=duration_min
                )
            
            self.set_cached_route(key, (distance_km, duration_min))
            return distance_km, duration_min
            
        except Exception as e:
            logger.error(
                "OSRM request failed",
                country=country,
                coordinates=f"({lat1},{lon1})→({lat2},{lon2})",
                error=str(e)
            )
            
            # Fallback to haversine distance
            dist = self.haversine_distance(lat1, lon1, lat2, lon2)
            duration = dist  # approx 1 min/km
            
            if Config.DEBUG:
                logger.debug(
                    "Using haversine fallback",
                    distance_km=dist,
                    duration_min=duration
                )
            
            return dist, duration
    
    def create_distance_time_matrices_parallel(
        self,
        locations: List[Dict[str, Any]]
    ) -> Tuple[List[List[float]], List[List[int]]]:
        """Create distance and time matrices using parallel processing"""
        size = len(locations)
        dist_matrix = np.zeros((size, size))
        time_matrix = np.zeros((size, size), dtype=int)

        def fetch(i: int, j: int) -> Tuple[int, int, float, float]:
            """Worker function for parallel distance calculation"""
            if i == j:
                return (i, j, 0, 0)
            
            country = locations[i]["Pays"]
            lat1, lon1 = locations[i]["Latitude"], locations[i]["Longitude"]
            lat2, lon2 = locations[j]["Latitude"], locations[j]["Longitude"]
            
            dist, duration = self.get_osrm_route_distance_duration(
                lat1, lon1, lat2, lon2, country
            )
            return (i, j, dist, duration)

        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            futures = {
                executor.submit(fetch, i, j): (i, j) 
                for i in range(size) 
                for j in range(size) 
                if i != j
            }
            
            for future in as_completed(futures):
                i, j, dist, duration = future.result()
                dist_matrix[i, j] = dist
                time_matrix[i, j] = int(round(duration))

        return dist_matrix.tolist(), time_matrix.tolist()
