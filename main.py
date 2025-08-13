# main.py
from datetime import datetime, date, timedelta, time
from typing import Dict, List, Tuple, Any
import pandas as pd
from config import Config
from app_logging import configure_logging, logger  # Un seul import de logging
from database import DatabaseManager
from routing import RouteCalculator
from optimization import VRPOptimizer
from clustering import LocationClusterer
from models import TrajectGenerator
from math import ceil
class VRPSolver:
    def __init__(self):
        self.db = DatabaseManager()
        self.route_calculator = RouteCalculator()
        self.optimizer = VRPOptimizer()
        self.clusterer = LocationClusterer()
        self.traject_generator = TrajectGenerator()
        
        self.driver_vehicles_cache: Dict[int, List[int]] = {}
    
    def get_vehicle_info(self) -> pd.DataFrame:
        """Get vehicle and driver information from database"""
        query = """
            SELECT v.Id as VehiculeId,
                   v.IdDriver as ChauffeurId,
                   a.DateAchatq
            FROM [fleet_management].[dbo].[Vehicules] v
            JOIN [fleet_management].[dbo].[AchatsVehicules] a ON v.Id = a.VehiculeId
            WHERE v.IdDriver IS NOT NULL AND a.DateAchatq IS NOT NULL
        """
        df = self.db.get_dataframe(query)
        df['DateAchatq'] = pd.to_datetime(df['DateAchatq']).dt.date
        logger.info("Vehicle info loaded", num_vehicles=len(df))
        return df
    
    def get_locations(self) -> List[Dict[str, Any]]:
        """Get locations from database"""
        query = """
            SELECT [Id], [Ville], [Pays], [Adresse], [Latitude], [Longitude]
            FROM [fleet_management].[dbo].[VillesAdresses]
            WHERE [Pays] IN ('Belgique', 'Pays-Bas', 'Luxembourg')
        """
        locations = self.db.execute_query(query)
        logger.info("Locations loaded", num_locations=len(locations))
        return locations
    
    def get_existing_durations(
        self, 
        start_date: date, 
        end_date: date
    ) -> Dict[Tuple[date, int], float]:
        """Get existing work durations for drivers"""
        query = """
            SELECT DateTrajet, ChauffeurId, SUM(DureeReelle) AS DureeTotale
            FROM [fleet_management].[dbo].[PlanningsTrajet]
            WHERE DateTrajet BETWEEN :start_date AND :end_date
            GROUP BY DateTrajet, ChauffeurId
        """
        params = {"start_date": start_date, "end_date": end_date}
        df = self.db.get_dataframe(query, params)
        
        durations = {
            (row["DateTrajet"].date(), row["ChauffeurId"]): row["DureeTotale"]
            for _, row in df.iterrows()
        }
        logger.info("Existing durations loaded", num_entries=len(durations))
        return durations
    
    def get_driver_vehicles(self, driver_id: int) -> List[int]:
        """Get vehicles assigned to a driver (with caching)"""
        if driver_id in self.driver_vehicles_cache:
            return self.driver_vehicles_cache[driver_id]
        
        query = """
            SELECT Id FROM [fleet_management].[dbo].[Vehicules]
            WHERE IdDriver = :driver_id
            ORDER BY Id
        """
        params = {"driver_id": int(driver_id)}
        vehicles = self.db.execute_query(query, params)
        vehicle_ids = [v["Id"] for v in vehicles]
        
        self.driver_vehicles_cache[driver_id] = vehicle_ids
        logger.info("Driver vehicles loaded", driver_id=driver_id, num_vehicles=len(vehicle_ids))
        return vehicle_ids
    
    def generate_optimized_trajects(
        self,
        vehicle_info_df: pd.DataFrame,
        locations: List[Dict[str, Any]],
        start_date: date,
        end_date: date,
        existing_durations: Dict[Tuple[date, int], float],
        chunk_size: int = Config.CHUNK_SIZE
    ) -> pd.DataFrame:
        """Generate optimized trajects for the given date range"""
        trajects = []
        current_date = start_date

        while current_date <= end_date:
            if current_date.weekday() == 6:  # Skip Sunday
                current_date += timedelta(days=1)
                continue

            for _, row in vehicle_info_df.iterrows():
                vehicle_id = row["VehiculeId"]
                driver_id = row["ChauffeurId"]
                purchase_date = row["DateAchatq"]
                
                # Skip if vehicle was recently purchased
                if current_date < (purchase_date + timedelta(days=3)):
                    continue
                
                # Check available work time
                duration_key = (current_date, driver_id)
                cumulative_duration = existing_durations.get(duration_key, 0)
                available_time = Config.MAX_DAILY_WORK_MINUTES - cumulative_duration
                
                if available_time <= 0:
                    continue
                
                # Get all vehicles for this driver
                driver_vehicles = self.get_driver_vehicles(driver_id)
                num_driver_vehicles = max(1, len(driver_vehicles))
                
                # Process location chunks
                for location_chunk in self.clusterer.chunk_locations(locations, chunk_size):
                    # Calculate distance and time matrices
                    dist_matrix, time_matrix = self.route_calculator.create_distance_time_matrices_parallel(location_chunk)
                    
                    # Solve VRP
                    routes = self.optimizer.solve_vrp(
                        dist_matrix,
                        time_matrix,
                        max_time=available_time,
                        num_vehicles=num_driver_vehicles,
                        depot=0  # Assuming first location is depot
                    )
                    
                    if not routes:
                        continue
                    
                    # Generate trajects from routes
                    if isinstance(routes[0], list):  # Multiple vehicles
                        for vehicle_idx, route in enumerate(routes):
                            last_time = None
                            for i in range(len(route)-1):
                                loc_idx = route[i]
                                next_idx = route[i+1]
                                pickup_loc = location_chunk[loc_idx]
                                depot_loc = location_chunk[next_idx]
                                
                                dist = dist_matrix[loc_idx][next_idx]
                                duration = time_matrix[loc_idx][next_idx]
                                
                                if last_time is None:
                                    pickup_time = datetime.combine(current_date, time(8, 30))
                                else:
                                    pickup_time = last_time + timedelta(minutes=40)
                                last_time = pickup_time
                                
                                trajects.append(
                                    self.traject_generator.generate_traject_dict(
                                        current_date,
                                        driver_vehicles[vehicle_idx],
                                        driver_id,
                                        pickup_loc,
                                        depot_loc,
                                        dist,
                                        duration,
                                        pickup_time
                                    )
                                )
                    else:  # Single vehicle
                        last_time = None
                        for i in range(len(routes)-1):
                            loc_idx = routes[i]
                            next_idx = routes[i+1]
                            pickup_loc = location_chunk[loc_idx]
                            depot_loc = location_chunk[next_idx]
                            
                            dist = dist_matrix[loc_idx][next_idx]
                            duration = time_matrix[loc_idx][next_idx]
                            
                            if last_time is None:
                                pickup_time = datetime.combine(current_date, time(8, 30))
                            else:
                                pickup_time = last_time + timedelta(minutes=40)
                            last_time = pickup_time
                            
                            trajects.append(
                                self.traject_generator.generate_traject_dict(
                                    current_date,
                                    vehicle_id,
                                    driver_id,
                                    pickup_loc,
                                    depot_loc,
                                    dist,
                                    duration,
                                    pickup_time
                                )
                            )

            current_date += timedelta(days=1)

        df = pd.DataFrame(trajects)
        logger.info("Trajects generated", num_trajects=len(df))
        return df
    
    def save_trajects(self, trajects_df: pd.DataFrame) -> None:
        """Save generated trajects to database"""
        if trajects_df.empty:
            logger.warning("No trajects to save")
            return
        
        # Format time columns
        trajects_df = trajects_df.copy()
        for col in ['HeureRetrait', 'HeureDepot']:
            trajects_df[col] = trajects_df[col].apply(
                lambda t: t.strftime("%H:%M:%S") if pd.notnull(t) else None
            )
        
        self.db.insert_dataframe(trajects_df, 'PlanningsTrajet')
    
    def run(self, start_date: date, end_date: date) -> None:
        """Main execution method"""
        start_time = datetime.now()
        logger.info("Starting VRP solver", start_date=start_date, end_date=end_date)
        
        try:
            # Load data
            vehicle_info_df = self.get_vehicle_info()
            locations = self.get_locations()
            existing_durations = self.get_existing_durations(start_date, end_date)
            
            # Generate trajects
            trajects_df = self.generate_optimized_trajects(
                vehicle_info_df,
                locations,
                start_date,
                end_date,
                existing_durations
            )
            
            # Save results
            self.save_trajects(trajects_df)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(
                "VRP solver completed successfully",
                duration_seconds=duration,
                num_trajects=len(trajects_df)
            )
            
        except Exception as e:
            logger.error("VRP solver failed", error=str(e))
            raise


if __name__ == "__main__":
    from app_logging import configure_logging
    configure_logging(Config.DEBUG)
    
    solver = VRPSolver()
    solver.run(date(2023, 1, 1), date.today())