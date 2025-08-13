# clustering.py
from sklearn.cluster import KMeans
import numpy as np
from typing import List, Dict, Any, Iterator
from config import Config
from custom_logging import logger

class LocationClusterer:
    def __init__(self):
        self.random_state = 42
    
    def chunk_locations(
        self, 
        locations: List[Dict[str, Any]], 
        chunk_size: int = Config.CHUNK_SIZE
    ) -> Iterator[List[Dict[str, Any]]]:
        """Cluster locations into chunks for parallel processing"""
        if len(locations) <= chunk_size:
            yield locations
            return

        coords = np.array([[loc["Latitude"], loc["Longitude"]] for loc in locations])
        n_clusters = max(1, ceil(len(locations) / chunk_size))

        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
        labels = kmeans.fit_predict(coords)

        for cluster_id in range(n_clusters):
            cluster_locations = [
                locations[i] 
                for i, lbl in enumerate(labels) 
                if lbl == cluster_id
            ]
            yield cluster_locations