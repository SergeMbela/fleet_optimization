# database.py
from sqlalchemy import create_engine, text
from sqlalchemy.engine.base import Engine
from contextlib import contextmanager
from typing import Iterator, List, Dict, Any
import pandas as pd
from config import Config
from exceptions import DatabaseError
from custom_logging import logger

class DatabaseManager:
    def __init__(self):
        self.engine = create_engine(
            Config.get_db_connection_string(), 
            fast_executemany=True,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_pre_ping=True
        )
    
    @contextmanager
    def get_connection(self) -> Iterator[Any]:
        """Context manager for database connections"""
        conn = None
        try:
            conn = self.engine.connect()
            yield conn
        except Exception as e:
            logger.error("Database connection failed", error=str(e))
            raise DatabaseError from e
        finally:
            if conn:
                conn.close()
    
    def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a SQL query and return results as dictionaries"""
        try:
            with self.get_connection() as conn:
                result = conn.execute(text(query), params or {})
                return [dict(row) for row in result]
        except Exception as e:
            logger.error("Query execution failed", query=query, error=str(e))
            raise DatabaseError from e
    
    def get_dataframe(self, query: str, params: Dict[str, Any] = None) -> pd.DataFrame:
        """Execute a SQL query and return results as a DataFrame"""
        try:
            with self.get_connection() as conn:
                return pd.read_sql(text(query), conn, params=params)
        except Exception as e:
            logger.error("DataFrame query failed", query=query, error=str(e))
            raise DatabaseError from e
    
    def insert_dataframe(self, df: pd.DataFrame, table: str, schema: str = "dbo") -> None:
        """Insert a DataFrame into the database"""
        if df.empty:
            logger.warning("Attempted to insert empty DataFrame", table=table)
            return
        
        try:
            with self.get_connection() as conn:
                df.to_sql(
                    table,
                    con=conn,
                    schema=schema,
                    if_exists='append',
                    index=False,
                    method='multi'
                )
            logger.info("Data inserted successfully", table=table, rows=len(df))
        except Exception as e:
            logger.error("Data insertion failed", table=table, error=str(e))
            raise DatabaseError from e