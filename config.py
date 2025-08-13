# config.py
import os
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()

class Config:
    # Database Configuration
    DB_DRIVER = os.getenv("DB_DRIVER")
    DB_SERVER = os.getenv("DB_SERVER")
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    
    # OSRM Configuration
    OSRM_SERVERS = {
        "Luxembourg": "http://localhost:5000",
        "Belgique": "http://localhost:5001",
        "Pays-Bas": "http://localhost:5002",
    }
    
    # Application Settings
    MAX_WORKERS = 10
    CHUNK_SIZE = 20
    MAX_DAILY_WORK_MINUTES = 510  # 8.5 hours
    OSRM_TIMEOUT = 5
    CACHE_TTL = 3600  # 1 hour
    DEBUG = True
    
    @classmethod
    def get_db_connection_string(cls) -> str:
        from urllib.parse import quote_plus
        driver_encoded = quote_plus(cls.DB_DRIVER)
        password_encoded = quote_plus(cls.DB_PASSWORD)
        return (
            f"mssql+pyodbc://{cls.DB_USER}:{password_encoded}@{cls.DB_SERVER}/{cls.DB_NAME}"
            f"?driver={driver_encoded}&TrustServerCertificate=yes"
        )