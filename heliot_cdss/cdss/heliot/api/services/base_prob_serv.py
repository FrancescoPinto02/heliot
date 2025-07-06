from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Union
from enum import Enum
import time
from dataclasses import dataclass

class LogprobsServiceType(Enum):
    """Tipi di servizi logprobs disponibili"""
    LOCAL_MODEL = "local_model"
    HEURISTIC = "heuristic"


@dataclass
class LogprobsResult:
    """Risultato standardizzato per calcolo logprobs"""
    logprobs: List[float]
    service_type: LogprobsServiceType
    computation_time: float
    token_count: int
    confidence_score: float
    metadata: Dict[str, Any]
    success: bool = True
    error_message: Optional[str] = None

@dataclass
class ServiceCapabilities:
    """Capacità del servizio logprobs"""
    supports_streaming: bool
    supports_batch: bool
    requires_internet: bool
    requires_gpu: bool
    max_content_length: Optional[int]
    supported_models: List[str]
    average_latency_ms: Optional[float]

class LogprobsService(ABC):
    """
    Classe astratta base per tutti i servizi di calcolo logprobs.
    Definisce l'interfaccia comune che tutti i servizi devono implementare.
    """
    
    def __init__(self, service_type: LogprobsServiceType, config: Optional[Dict] = None):
        self.service_type = service_type
        self.config = config or {}
        self._initialized = False
        self._last_error = None
        self._call_count = 0
        self._total_computation_time = 0.0
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Inizializza il servizio (caricamento modelli, setup client, etc.)
        
        Returns:
            bool: True se inizializzazione riuscita, False altrimenti
        """
        pass
    
    @abstractmethod
    def calculate_streaming_logprobs(self, system_prompt: str, user_prompt: str, 
                                   accumulated_content: str, **kwargs) -> LogprobsResult:
        """
        Calcola logprobs per contenuto accumulato in streaming
        
        Args:
            system_prompt: Prompt di sistema
            user_prompt: Prompt utente
            accumulated_content: Contenuto accumulato fino ad ora
            **kwargs: Parametri aggiuntivi specifici del servizio
            
        Returns:
            LogprobsResult: Risultato con logprobs e metadata
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> ServiceCapabilities:
        """
        Restituisce le capacità del servizio
        
        Returns:
            ServiceCapabilities: Capacità del servizio
        """
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """
        Controlla se il servizio è funzionante
        
        Returns:
            bool: True se il servizio è disponibile
        """
        pass
    
    def calculate_batch_logprobs(self, requests: List[Dict[str, str]], **kwargs) -> List[LogprobsResult]:
        """
        Calcola logprobs per più richieste in batch (implementazione di default)
        
        Args:
            requests: Lista di richieste, ogni dict deve contenere system_prompt, user_prompt, content
            **kwargs: Parametri aggiuntivi
            
        Returns:
            List[LogprobsResult]: Lista di risultati
        """
        results = []
        for request in requests:
            try:
                result = self.calculate_streaming_logprobs(
                    request.get('system_prompt', ''),
                    request.get('user_prompt', ''),
                    request.get('content', ''),
                    **kwargs
                )
                results.append(result)
            except Exception as e:
                error_result = LogprobsResult(
                    logprobs=[],
                    service_type=self.service_type,
                    computation_time=0.0,
                    token_count=0,
                    confidence_score=0.0,
                    metadata={'error': str(e)},
                    success=False,
                    error_message=str(e)
                )
                results.append(error_result)
        
        return results
    
    def validate_input(self, system_prompt: str, user_prompt: str, content: str) -> bool:
        """
        Valida input prima del calcolo
        
        Returns:
            bool: True se input valido
        """
        if not content or not content.strip():
            self._last_error = "Content is empty"
            return False
        
        capabilities = self.get_capabilities()
        if capabilities.max_content_length and len(content) > capabilities.max_content_length:
            self._last_error = f"Content too long: {len(content)} > {capabilities.max_content_length}"
            return False
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Restituisce statistiche di utilizzo del servizio"""
        avg_time = self._total_computation_time / max(1, self._call_count)
        
        return {
            'service_type': self.service_type.value,
            'initialized': self._initialized,
            'call_count': self._call_count,
            'total_computation_time': self._total_computation_time,
            'average_computation_time': avg_time,
            'last_error': self._last_error,
            'health_status': self.health_check()
        }
    
    def reset_statistics(self):
        """Reset delle statistiche"""
        self._call_count = 0
        self._total_computation_time = 0.0
        self._last_error = None
    
    def _record_call(self, computation_time: float, success: bool = True, error: Optional[str] = None):
        """Registra una chiamata per statistiche"""
        self._call_count += 1
        self._total_computation_time += computation_time
        if not success:
            self._last_error = error
    
    def _measure_time(self, func, *args, **kwargs):
        """Utility per misurare tempo di esecuzione"""
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
            raise
        finally:
            computation_time = time.time() - start_time
            self._record_call(computation_time, success, error)
        
        return result, computation_time
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.service_type.value})"
    
    def __repr__(self) -> str:
        return self.__str__()
