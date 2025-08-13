# exceptions.py
class RoutingError(Exception):
    """Base exception for routing-related errors"""
    pass

class OSRMError(RoutingError):
    """Exception for OSRM-related errors"""
    pass

class DatabaseError(RoutingError):
    """Exception for database-related errors"""
    pass

class OptimizationError(RoutingError):
    """Exception for optimization failures"""
    pass