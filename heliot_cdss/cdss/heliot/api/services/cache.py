import time
import threading
import hashlib
import json
import pickle
import os
import atexit
import signal
import sys
from pathlib import Path
from typing import Any, Optional, Callable, Union, Tuple, List, Dict, Hashable
from dataclasses import dataclass, asdict

@dataclass
class CacheEntry:
    """Rappresenta un elemento in cache con il suo timestamp di scadenza"""
    value: Any
    expires_at: float
    
    def to_dict(self) -> dict:
        """Converte in dizionario per serializzazione JSON (per tipi semplici)"""
        return {
            'value': self.value,
            'expires_at': self.expires_at
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CacheEntry':
        """Crea CacheEntry da dizionario"""
        return cls(value=data['value'], expires_at=data['expires_at'])

class CompositeKey:
    """
    Classe per gestire chiavi composite di qualsiasi tipo.
    Genera automaticamente una chiave hashable da componenti arbitrari.
    """
    
    def __init__(self, *components):
        """
        Crea una chiave composite dai componenti forniti.
        
        Args:
            *components: Componenti della chiave (stringhe, liste, dict, numeri, etc.)
        """
        self.components = components
        self._hash = None
        self._key_string = None
    
    def _serialize_component(self, component) -> str:
        """Serializza un singolo componente in stringa deterministica."""
        if isinstance(component, str):
            return f"str:{component}"
        elif isinstance(component, (int, float)):
            return f"num:{component}"
        elif isinstance(component, bool):
            return f"bool:{component}"
        elif isinstance(component, (list, tuple)):
            # Ordina se contiene elementi hashable, altrimenti mantiene l'ordine
            try:
                if all(isinstance(x, (str, int, float, bool)) for x in component):
                    sorted_items = sorted(component)
                    return f"list:{json.dumps(sorted_items, sort_keys=True)}"
                else:
                    return f"list:{json.dumps(component, sort_keys=True, default=str)}"
            except:
                return f"list:{str(component)}"
        elif isinstance(component, dict):
            return f"dict:{json.dumps(component, sort_keys=True, default=str)}"
        elif isinstance(component, set):
            return f"set:{json.dumps(sorted(list(component)), sort_keys=True, default=str)}"
        else:
            # Per oggetti custom usa la rappresentazione stringa
            return f"obj:{str(component)}"
    
    def to_string(self) -> str:
        """Converte la chiave in stringa deterministica."""
        if self._key_string is None:
            serialized_components = [
                self._serialize_component(comp) for comp in self.components
            ]
            self._key_string = "|".join(serialized_components)
        return self._key_string
    
    def to_hash(self) -> str:
        """Converte la chiave in hash SHA256 per chiavi molto lunghe."""
        if self._hash is None:
            key_string = self.to_string()
            self._hash = hashlib.sha256(key_string.encode()).hexdigest()
        return self._hash
    
    def __str__(self) -> str:
        return self.to_string()
    
    def __hash__(self) -> int:
        return hash(self.to_string())
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, CompositeKey):
            return False
        return self.to_string() == other.to_string()

# Registro globale delle cache per gestione dei segnali
_cache_registry = []
_signal_handlers_installed = False

def _install_signal_handlers():
    """Installa i gestori di segnale una sola volta"""
    global _signal_handlers_installed
    if _signal_handlers_installed:
        return
    
    def signal_handler(signum, frame):
        """Gestisce i segnali salvando tutte le cache registrate"""
        print(f"\nRicevuto segnale {signum}, salvataggio cache in corso...")
        
        for cache in _cache_registry:
            try:
                if hasattr(cache, '_save_cache') and cache._auto_persist:
                    cache._save_cache()
            except Exception as e:
                print(f"Errore nel salvataggio cache: {e}")
        
        print("Salvataggio cache completato.")
        
        # Ripristina il comportamento predefinito e rilancia il segnale
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    
    # Registra gestori per i segnali comuni
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Terminazione normale
    
    # Su Unix registra anche altri segnali
    if hasattr(signal, 'SIGHUP'):
        signal.signal(signal.SIGHUP, signal_handler)   # Hangup
    if hasattr(signal, 'SIGQUIT'):
        signal.signal(signal.SIGQUIT, signal_handler)  # Ctrl+\
    
    _signal_handlers_installed = True

class ThreadSafeCache:
    """
    Cache thread-safe che supporta chiavi composite e persistenza robusta.
    """
    
    def __init__(self, default_ttl: float = 300, use_hash_keys: bool = False, 
                 cache_file: Optional[str] = None, auto_persist: bool = True,
                 persist_format: str = 'pickle', periodic_save: bool = False,
                 save_interval: float = 300.0):
        """
        Inizializza la cache.
        
        Args:
            default_ttl: Durata di vita predefinita degli oggetti in secondi
            use_hash_keys: Se True usa hash SHA256 per chiavi molto lunghe
            cache_file: Percorso del file per la persistenza (default: ./cache_data)
            auto_persist: Se True salva automaticamente alla distruzione
            persist_format: Formato di serializzazione ('pickle' o 'json')
            periodic_save: Se True salva periodicamente in background
            save_interval: Intervallo in secondi per il salvataggio periodico
        """
        self._cache = {}
        self._default_ttl = default_ttl
        self._use_hash_keys = use_hash_keys
        self._local = threading.local()
        
        # Configurazione persistenza
        self._cache_file = cache_file or "./cache_data"
        self._auto_persist = auto_persist
        self._persist_format = persist_format.lower()
        self._periodic_save = periodic_save
        self._save_interval = save_interval
        self._last_save_time = time.time()
        self._save_lock = threading.RLock()
        
        if self._persist_format not in ['pickle', 'json']:
            raise ValueError("persist_format deve essere 'pickle' o 'json'")
        
        # Registra questa cache per la gestione dei segnali
        global _cache_registry
        _cache_registry.append(self)
        _install_signal_handlers()
        
        # Carica cache esistente
        self._load_cache()
        
        # Registra il salvataggio automatico alla chiusura del programma
        if self._auto_persist:
            atexit.register(self._save_cache)
        
        # Avvia thread per salvataggio periodico se richiesto
        if self._periodic_save:
            self._start_periodic_save_thread()
    
    def _start_periodic_save_thread(self):
        """Avvia thread per salvataggio periodico in background"""
        def periodic_save():
            while getattr(self, '_periodic_save', False):
                try:
                    time.sleep(self._save_interval)
                    if getattr(self, '_periodic_save', False):  # Controlla di nuovo
                        current_time = time.time()
                        if current_time - self._last_save_time >= self._save_interval:
                            self._save_cache()
                except Exception as e:
                    print(f"Errore nel salvataggio periodico: {e}")
        
        thread = threading.Thread(target=periodic_save, daemon=True)
        thread.start()
    
    def _get_cache_file_path(self) -> str:
        """Restituisce il percorso completo del file cache con estensione appropriata"""
        base_path = self._cache_file
        if self._persist_format == 'pickle':
            return f"{base_path}.pkl"
        else:
            return f"{base_path}.json"
    
    def _load_cache(self) -> None:
        """Carica la cache dal file system se esiste."""
        cache_path = self._get_cache_file_path()
        
        if not os.path.exists(cache_path):
            return
        
        try:
            with self._save_lock:
                current_time = time.time()
                loaded_count = 0
                expired_count = 0
                
                if self._persist_format == 'pickle':
                    with open(cache_path, 'rb') as f:
                        cache_data = pickle.load(f)
                else:  # json
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                
                # Ricostruisce la cache filtrando gli elementi scaduti
                for key, entry_data in cache_data.items():
                    if self._persist_format == 'pickle':
                        entry = entry_data  # CacheEntry è già deserializzato
                    else:  # json
                        entry = CacheEntry.from_dict(entry_data)
                    
                    if current_time <= entry.expires_at:
                        self._cache[key] = entry
                        loaded_count += 1
                    else:
                        expired_count += 1
                
                print(f"Cache caricata: {loaded_count} elementi validi, {expired_count} scaduti ignorati")
                
        except Exception as e:
            print(f"Errore nel caricamento della cache da {cache_path}: {e}")
            self._cache = {}
    
    def _save_cache(self) -> None:
        """Salva la cache corrente su file system."""
        if not self._cache:
            return
        
        cache_path = self._get_cache_file_path()
        
        try:
            with self._save_lock:
                # Crea la directory se non esiste
                os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
                
                # Filtra gli elementi scaduti prima di salvare
                current_time = time.time()
                valid_cache = {}
                
                for key, entry in self._cache.items():
                    if current_time <= entry.expires_at:
                        valid_cache[key] = entry
                
                # Salva in un file temporaneo prima per atomicità
                temp_path = f"{cache_path}.tmp"
                
                if self._persist_format == 'pickle':
                    with open(temp_path, 'wb') as f:
                        pickle.dump(valid_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
                else:  # json
                    # Per JSON, converte CacheEntry in dict
                    json_cache = {}
                    for key, entry in valid_cache.items():
                        try:
                            # Prova a serializzare il valore per verificare compatibilità JSON
                            json.dumps(entry.value)
                            json_cache[key] = entry.to_dict()
                        except (TypeError, ValueError):
                            # Salta valori non serializzabili in JSON
                            continue
                    
                    with open(temp_path, 'w', encoding='utf-8') as f:
                        json.dump(json_cache, f, indent=2, ensure_ascii=False)
                
                # Sposta atomicamente il file temporaneo
                if os.path.exists(temp_path):
                    if os.path.exists(cache_path):
                        os.replace(temp_path, cache_path)
                    else:
                        os.rename(temp_path, cache_path)
                
                self._last_save_time = time.time()
                print(f"Cache salvata: {len(valid_cache)} elementi in {cache_path}")
                
        except Exception as e:
            print(f"Errore nel salvataggio della cache in {cache_path}: {e}")
            # Rimuovi file temporaneo se esiste
            temp_path = f"{cache_path}.tmp"
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
    
    def save_to_file(self, file_path: Optional[str] = None) -> bool:
        """
        Salva manualmente la cache su file.
        
        Args:
            file_path: Percorso personalizzato (opzionale)
            
        Returns:
            True se il salvataggio è riuscito, False altrimenti
        """
        if file_path:
            original_path = self._cache_file
            self._cache_file = file_path
            
        try:
            self._save_cache()
            return True
        except:
            return False
        finally:
            if file_path:
                self._cache_file = original_path
    
    def load_from_file(self, file_path: str, merge: bool = False) -> bool:
        """
        Carica la cache da un file specifico.
        
        Args:
            file_path: Percorso del file da caricare
            merge: Se True unisce con la cache esistente, altrimenti la sostituisce
            
        Returns:
            True se il caricamento è riuscito, False altrimenti
        """
        if not merge:
            self._cache = {}
        
        original_path = self._cache_file
        self._cache_file = file_path
        
        try:
            self._load_cache()
            return True
        except:
            return False
        finally:
            self._cache_file = original_path
    
    def _normalize_key(self, key) -> str:
        """Normalizza la chiave in stringa."""
        if isinstance(key, str):
            return key
        elif isinstance(key, CompositeKey):
            return key.to_hash() if self._use_hash_keys else key.to_string()
        elif isinstance(key, (tuple, list)):
            composite_key = CompositeKey(*key)
            return composite_key.to_hash() if self._use_hash_keys else composite_key.to_string()
        else:
            composite_key = CompositeKey(key)
            return composite_key.to_hash() if self._use_hash_keys else composite_key.to_string()
    
    def get(self, key) -> Optional[Any]:
        """Recupera un valore dalla cache."""
        normalized_key = self._normalize_key(key)
        entry = self._cache.get(normalized_key)
        
        if entry is None:
            return None
            
        current_time = time.time()
        if current_time > entry.expires_at:
            self._cache.pop(normalized_key, None)
            return None
            
        return entry.value
    
    def put(self, key, value: Any, ttl: Optional[float] = None) -> None:
        """Inserisce un valore nella cache."""
        if ttl is None:
            ttl = self._default_ttl
            
        normalized_key = self._normalize_key(key)
        expires_at = time.time() + ttl
        entry = CacheEntry(value=value, expires_at=expires_at)
        
        self._cache[normalized_key] = entry
    
    def delete(self, key) -> bool:
        """Rimuove un elemento dalla cache."""
        normalized_key = self._normalize_key(key)
        return self._cache.pop(normalized_key, None) is not None
    
    def has(self, key) -> bool:
        """Verifica se una chiave esiste nella cache ed è valida."""
        return self.get(key) is not None
    
    def clear(self) -> None:
        """Svuota completamente la cache."""
        self._cache = {}
    
    def cleanup_expired(self) -> int:
        """Rimuove tutti gli elementi scaduti dalla cache."""
        current_time = time.time()
        new_cache = {}
        removed_count = 0
        
        for key, entry in self._cache.items():
            if current_time <= entry.expires_at:
                new_cache[key] = entry
            else:
                removed_count += 1
        
        self._cache = new_cache
        return removed_count
    
    def size(self) -> int:
        """Restituisce il numero di elementi nella cache."""
        return len(self._cache)
    
    def get_or_compute(self, key, compute_func: Callable[[], Any], 
                      ttl: Optional[float] = None) -> Any:
        """Recupera un valore dalla cache o lo calcola se non presente/scaduto."""
        value = self.get(key)
        if value is not None:
            return value
        
        computed_value = compute_func()
        self.put(key, computed_value, ttl)
        return computed_value
    
    def stats(self) -> dict:
        """Restituisce statistiche sulla cache."""
        current_time = time.time()
        total_items = len(self._cache)
        expired_items = sum(1 for entry in self._cache.values() 
                          if current_time > entry.expires_at)
        valid_items = total_items - expired_items
        
        return {
            'total_items': total_items,
            'valid_items': valid_items,
            'expired_items': expired_items,
            'default_ttl': self._default_ttl,
            'use_hash_keys': self._use_hash_keys,
            'cache_file': self._get_cache_file_path(),
            'persist_format': self._persist_format,
            'periodic_save': self._periodic_save,
            'last_save_time': self._last_save_time
        }
    
    def __del__(self):
        """Salva automaticamente la cache quando l'oggetto viene distrutto."""
        if hasattr(self, '_auto_persist') and self._auto_persist:
            try:
                # Disabilita salvataggio periodico
                self._periodic_save = False
                self._save_cache()
                
                # Rimuovi dal registro globale
                global _cache_registry
                if self in _cache_registry:
                    _cache_registry.remove(self)
            except:
                pass

# Funzioni helper
def make_key(*components) -> CompositeKey:
    """Helper per creare chiavi composite."""
    return CompositeKey(*components)

# Esempio di utilizzo
if __name__ == "__main__":
    print("=== Test Cache con gestione segnali ===")
    print("Prova a premere Ctrl+C durante l'esecuzione per testare il salvataggio...")
    
    # Cache con salvataggio periodico ogni 5 secondi
    cache = ThreadSafeCache(
        default_ttl=3600,
        cache_file="test_robust_cache",
        periodic_save=True,
        save_interval=5.0
    )
    
    # Aggiungi dati
    for i in range(10):
        cache.put(f"key_{i}", f"value_{i}", ttl=7200)
        cache.put(("composite", i), {"data": f"complex_data_{i}"}, ttl=7200)
    
    print(f"Cache popolata: {cache.stats()}")
    print("Cache in esecuzione... (premi Ctrl+C per testare il salvataggio)")
    
    try:
        # Simula lavoro continuo
        for i in range(60):  # 1 minuto
            time.sleep(1)
            if i % 10 == 0:
                cache.put(f"dynamic_key_{i}", f"dynamic_value_{i}")
                print(f"Aggiunto dynamic_key_{i}")
    
    except KeyboardInterrupt:
        print("\nInterruzione ricevuta!")
    
    print("Programma terminato.")