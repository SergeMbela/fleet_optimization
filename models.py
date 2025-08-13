# models.py
from datetime import datetime, time, date, timedelta
from typing import Dict, Any
import uuid
from config import Config
from custom_logging import logger

class TrajectGenerator:
    @staticmethod
    def calculate_depot_time(pickup_time: time, duration_minutes: float) -> time:
        """Calculate depot arrival time based on pickup time and duration"""
        pickup_dt = datetime.combine(date.today(), pickup_time)
        depot_dt = pickup_dt + timedelta(minutes=round(duration_minutes))
        return depot_dt.time()
    
    @staticmethod
    def format_duration(duration_minutes: float) -> str:
        """Format duration in minutes to human-readable string"""
        hours, minutes = divmod(duration_minutes, 60)
        if hours and minutes:
            return f"{hours}h {minutes:02d}min"
        if hours:
            return f"{hours}h"
        return f"{minutes}min"
    
    def generate_traject_dict(
        self,
        current_date: date,
        vehicle_id: int,
        driver_id: int,
        pickup_location: Dict[str, Any],
        depot_location: Dict[str, Any],
        distance: float,
        duration: float,
        pickup_time_dt: datetime
    ) -> Dict[str, Any]:
        """Generate a traject dictionary for database insertion"""
        pickup_time = pickup_time_dt.time()
        depot_time = self.calculate_depot_time(pickup_time, duration)
        formatted_duration = self.format_duration(duration)
        
        return {
            "Uuid": str(uuid.uuid4()),
            "DateTrajet": current_date,
            "VehiculeId": vehicle_id,
            "ChauffeurId": driver_id,
            "LieuRetrait": pickup_location["Ville"],
            "LieuDepot": depot_location["Ville"],
            "DistanceEstimee": round(distance, 2),
            "DureeEstimee": duration,
            "DureeReelle": duration,
            "Latitude": pickup_location["Latitude"],
            "Longitude": pickup_location["Longitude"],
            "HeureRetrait": pickup_time,
            "HeureDepot": depot_time,
            "DureeFormatee": formatted_duration,
            "CreatedAt": datetime.now()
        }