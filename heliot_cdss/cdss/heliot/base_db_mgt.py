from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseDatabaseManagement(ABC):
    """
    Abstract base class for database management operations.
    
    This class provides a common interface for database operations
    and allows configuration to be passed to concrete implementations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the database management instance.
        
        Args:
            config (Optional[Dict[str, Any]]): Configuration dictionary
                that can contain database connection parameters, credentials,
                or any other settings needed by concrete implementations.
        """
        self.config = config or {}
    
    @abstractmethod
    def search_drug(self, drug_code: str) -> Any:
        """
        Search for a drug in the database using its code.
        
        Args:
            drug_code (str): The unique code identifier for the drug
            
        Returns:
            Any: The drug data or result from the database search.
                 The specific return type depends on the concrete implementation.
                 
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        pass