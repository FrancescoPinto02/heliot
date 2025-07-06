from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
import json
import re
import numpy as np
import os
from openai import OpenAI
from .base_prob_serv import *

#
# Unifier Adapter to handle LLM responses
#
class UnifiedAdapter(ABC):
    
    def __init__(self):
        pass

    #
    # Indicate if the adapter supports native logprobs
    #  
    def supports_native_logprobs(self) -> bool:
        return True
    
    @abstractmethod
    def supports_native_usage_streaming(self) -> bool:
        pass

    @abstractmethod
    def extract_content_streaming(self, event) -> Optional[str]:
        pass
    
    @abstractmethod
    def extract_logprobs_streaming(self, event) -> List[float]:
        pass
    
    @abstractmethod
    def extract_usage_streaming(self, event) -> Optional[Dict]:
        pass
    
    @abstractmethod
    def is_done_streaming(self, event) -> bool:
        pass
    
    @abstractmethod
    def extract_content_non_streaming(self, response) -> str:
        pass
    
    @abstractmethod
    def extract_usage_non_streaming(self, response) -> Dict:
        pass

#
# OpenAI adapter
#
class OpenAIUnifiedAdapter(UnifiedAdapter):
    def supports_native_logprobs(self) -> bool:
        return True
    
    def extract_content_streaming(self, event) -> Optional[str]:
        if (hasattr(event, 'choices') and event.choices and 
            len(event.choices) > 0 and 
            hasattr(event.choices[0], 'delta') and
            hasattr(event.choices[0].delta, 'content')):
            return event.choices[0].delta.content
        return None
    
    def supports_native_usage_streaming(self) -> bool:
        return True

    def extract_logprobs_streaming(self, event) -> List[float]:
        logprobs = []
        if (hasattr(event, 'choices') and event.choices and 
            len(event.choices) > 0 and 
            hasattr(event.choices[0], 'logprobs') and
            event.choices[0].logprobs and
            hasattr(event.choices[0].logprobs, 'content')):
            for log_item in event.choices[0].logprobs.content:
                if hasattr(log_item, 'logprob'):
                    logprobs.append(log_item.logprob)
        return logprobs
    
    def extract_usage_streaming(self, event) -> Optional[Dict]:
        if hasattr(event, 'usage') and event.usage:
            return {
                'prompt_tokens': getattr(event.usage, 'prompt_tokens', 0),
                'completion_tokens': getattr(event.usage, 'completion_tokens', 0),
                'total_tokens': getattr(event.usage, 'total_tokens', 0)
            }
        return None
    
    def is_done_streaming(self, event) -> bool:
        return hasattr(event, 'usage') and event.usage is not None
    
    def extract_content_non_streaming(self, response) -> str:
        if (hasattr(response, 'choices') and response.choices and
            len(response.choices) > 0 and
            hasattr(response.choices[0], 'message') and
            hasattr(response.choices[0].message, 'content')):
            return response.choices[0].message.content or ""
        return ""
    
    def extract_usage_non_streaming(self, response) -> Dict:
        if hasattr(response, 'usage') and response.usage:
            return {
                'prompt_tokens': getattr(response.usage, 'prompt_tokens', 0),
                'completion_tokens': getattr(response.usage, 'completion_tokens', 0),
                'total_tokens': getattr(response.usage, 'total_tokens', 0)
            }
        return {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}

#
# OpenLLM adapter
#
class OpenLLMUnifiedAdapter(UnifiedAdapter):
    """Adapter per OpenLLM (compatibile OpenAI ma senza logprobs in streaming)"""
    
    def extract_content_streaming(self, event) -> Optional[str]:
        if (hasattr(event, 'choices') and event.choices and 
            len(event.choices) > 0 and 
            hasattr(event.choices[0], 'delta') and
            hasattr(event.choices[0].delta, 'content')):
            return event.choices[0].delta.content
        return None

    def supports_native_usage_streaming(self) -> bool:
        return False
    
    def extract_logprobs_streaming(self, event) -> List[float]:
        # OpenLLM non fornisce logprobs in streaming
        logprobs = []
        if (hasattr(event, 'choices') and event.choices and 
            len(event.choices) > 0 and 
            hasattr(event.choices[0], 'logprobs') and
            event.choices[0].logprobs and
            hasattr(event.choices[0].logprobs, 'content')):
            for log_item in event.choices[0].logprobs.content:
                if hasattr(log_item, 'logprob'):
                    logprobs.append(log_item.logprob)
        return logprobs
    
    def extract_usage_streaming(self, event) -> Optional[Dict]:
        if hasattr(event, 'usage') and event.usage:
            return {
                'prompt_tokens': getattr(event.usage, 'prompt_tokens', 0),
                'completion_tokens': getattr(event.usage, 'completion_tokens', 0),
                'total_tokens': getattr(event.usage, 'total_tokens', 0)
            }
        return None
    
    def is_done_streaming(self, event) -> bool:
        # Controlla finish_reason='stop' invece di usage
        if (hasattr(event, 'choices') and event.choices and 
            len(event.choices) > 0 and 
            hasattr(event.choices[0], 'finish_reason')):
            return event.choices[0].finish_reason == 'stop'
        return False
    
    def extract_content_non_streaming(self, response) -> str:
        if (hasattr(response, 'choices') and response.choices and
            len(response.choices) > 0 and
            hasattr(response.choices[0], 'message') and
            hasattr(response.choices[0].message, 'content')):
            return response.choices[0].message.content or ""
        return ""
    
    def extract_logprobs_non_streaming(self, response) -> List[float]:
        # Non calcoliamo logprobs per chiamate non-streaming
        return []
    
    def extract_usage_non_streaming(self, response) -> Dict:
        if hasattr(response, 'usage') and response.usage:
            return {
                'prompt_tokens': getattr(response.usage, 'prompt_tokens', 0),
                'completion_tokens': getattr(response.usage, 'completion_tokens', 0),
                'total_tokens': getattr(response.usage, 'total_tokens', 0)
            }
        return {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
#
# Anthropic Adapter
#
class AnthropicUnifiedAdapter(UnifiedAdapter):
    # Metodi streaming
    def extract_content_streaming(self, event) -> Optional[str]:
        if hasattr(event, 'delta') and hasattr(event.delta, 'text'):
            return event.delta.text
        elif hasattr(event, 'text'):
            return event.text
        return None
    
    def extract_logprobs_streaming(self, event) -> List[float]:
        return []  # Anthropic non fornisce logprobs nativamente
    
    def supports_native_usage_streaming(self) -> bool:
        return False
    
    def extract_usage_streaming(self, event) -> Optional[Dict]:
        if hasattr(event, 'usage'):
            input_tokens = getattr(event.usage, 'input_tokens', 0) or 0
            output_tokens = getattr(event.usage, 'output_tokens', 0) or 0
            
            return {
                'prompt_tokens': input_tokens,
                'completion_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens
            }
        return None
    
    def is_done_streaming(self, event) -> bool:
        return hasattr(event, 'type') and event.type == 'message_stop'
    
    # Metodi non-streaming
    def extract_content_non_streaming(self, response) -> str:
        if (hasattr(response, 'content') and response.content and
            len(response.content) > 0 and
            hasattr(response.content[0], 'text')):
            return response.content[0].text
        return ""
    
    def extract_logprobs_non_streaming(self, response) -> List[float]:
        return []  # Non calcoliamo logprobs per chiamate non-streaming
    
    def extract_usage_non_streaming(self, response) -> Dict:
        if hasattr(response, 'usage'):
            return {
                'prompt_tokens': getattr(response.usage, 'input_tokens', 0),
                'completion_tokens': getattr(response.usage, 'output_tokens', 0),
                'total_tokens': getattr(response.usage, 'input_tokens', 0) + getattr(response.usage, 'output_tokens', 0)
            }
        return {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    
#
# Gemini Adapter
#
class GeminiUnifiedAdapter(UnifiedAdapter):
    # Metodi streaming
    def extract_content_streaming(self, event) -> Optional[str]:
        if (hasattr(event, 'candidates') and event.candidates and
            len(event.candidates) > 0 and
            hasattr(event.candidates[0], 'content') and
            hasattr(event.candidates[0].content, 'parts')):
            parts = event.candidates[0].content.parts
            if parts and hasattr(parts[0], 'text'):
                return parts[0].text
        return None
    
    def extract_logprobs_streaming(self, event) -> List[float]:
        return []  # Gemini non fornisce logprobs nativamente

    def supports_native_usage_streaming(self) -> bool:
        return False
    
    def extract_usage_streaming(self, event) -> Optional[Dict]:
        if hasattr(event, 'usage_metadata'):
            return {
                'prompt_tokens': getattr(event.usage_metadata, 'prompt_token_count', 0),
                'completion_tokens': getattr(event.usage_metadata, 'candidates_token_count', 0),
                'total_tokens': getattr(event.usage_metadata, 'total_token_count', 0)
            }
        return None
    
    def is_done_streaming(self, event) -> bool:
        return (hasattr(event, 'candidates') and event.candidates and
                len(event.candidates) > 0 and
                hasattr(event.candidates[0], 'finish_reason') and
                event.candidates[0].finish_reason)
    
    # Metodi non-streaming
    def extract_content_non_streaming(self, response) -> str:
        if (hasattr(response, 'candidates') and response.candidates and
            len(response.candidates) > 0 and
            hasattr(response.candidates[0], 'content') and
            hasattr(response.candidates[0].content, 'parts')):
            parts = response.candidates[0].content.parts
            if parts and hasattr(parts[0], 'text'):
                return parts[0].text
        return ""
    
    def extract_logprobs_non_streaming(self, response) -> List[float]:
        return []  # Non calcoliamo logprobs per chiamate non-streaming
    
    def extract_usage_non_streaming(self, response) -> Dict:
        if hasattr(response, 'usage_metadata'):
            return {
                'prompt_tokens': getattr(response.usage_metadata, 'prompt_token_count', 0),
                'completion_tokens': getattr(response.usage_metadata, 'candidates_token_count', 0),
                'total_tokens': getattr(response.usage_metadata, 'total_token_count', 0)
            }
        return {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}

#
# Streamed JSON response parser
#
class JSONStreamParser:
    def __init__(self, expected_fields: List[str]):
        self.expected_fields = expected_fields
        self.reset()
    
    def reset(self):
        """Resetta lo stato del parser"""
        self.buffer = ""
        self.current_field = None
        self.field_index = -1
        self.in_value = False
        self.brace_count = 0
        self.quote_count = 0
        self.field_logprobs = {field: [] for field in self.expected_fields}
        self.all_logprobs = []
        self.field_positions = {}
        self.json_complete = False
    
    def process_chunk(self, content: str, logprobs: List[float] = None) -> Dict:
        """Processa un chunk di contenuto e restituisce info sul parsing"""
        if logprobs is None:
            logprobs = []
        
        result = {
            'field_changed': False,
            'previous_field': self.current_field,
            'current_field': self.current_field,
            'in_value': self.in_value,
            'json_complete': False,
            'chunk_length': len(content)
        }
        
        self.buffer += content
        self.all_logprobs.extend(logprobs)
        
        # Rileva cambio di campo
        previous_field = self.current_field
        for i, field in enumerate(self.expected_fields):
            pattern = f'"{field}"\\s*:'
            if re.search(pattern, content):
                result['previous_field'] = self.current_field
                self.current_field = field
                self.field_index = i
                result['field_changed'] = True
                result['current_field'] = field
                
                # Salva posizione del campo
                self.field_positions[field] = len(self.buffer) - len(content)
                break
        
        # Rileva se siamo nel valore
        if ':"' in content:
            self.in_value = True
            result['in_value'] = True
        elif '","' in content or '"}' in content:
            self.in_value = False
            result['in_value'] = False
        
        # Salva logprobs per il campo corrente
        if self.in_value and self.current_field and logprobs:
            self.field_logprobs[self.current_field].extend(logprobs)
        
        # Controlla se il JSON è completo
        brace_count_change = content.count('{') - content.count('}')
        self.brace_count += brace_count_change
        
        if '}' in content and self.brace_count <= 0:
            self.json_complete = True
            result['json_complete'] = True
        
        return result
    
    def get_parsing_stats(self) -> Dict:
        """Restituisce statistiche del parsing"""
        return {
            'total_content_length': len(self.buffer),
            'total_logprobs': len(self.all_logprobs),
            'fields_found': list(self.field_positions.keys()),
            'field_logprobs_count': {field: len(logprobs) for field, logprobs in self.field_logprobs.items()},
            'json_complete': self.json_complete,
            'current_field': self.current_field
        }

class JSONAnalyzer:
    """Analizza JSON completo per estrarre logprobs per campo"""
    
    def __init__(self, expected_fields: List[str]):
        self.expected_fields = expected_fields
    
    def analyze_json_content(self, content: str, all_logprobs: List[float]) -> Dict[str, List[float]]:
        """Analizza il contenuto JSON e assegna logprobs ai campi"""
        field_logprobs = {field: [] for field in self.expected_fields}
        
        if not all_logprobs or not content.strip():
            return field_logprobs
        
        try:
            # Trova le posizioni dei campi nel testo
            field_ranges = self._find_field_ranges(content)
            

            if not field_ranges:
                # Se non riusciamo a parsare, distribuiamo uniformemente
                return self._distribute_logprobs_uniformly(all_logprobs)
            
            # Distribuisce i logprobs in base alle posizioni dei caratteri
            content_length = len(content)
            logprobs_per_char = len(all_logprobs) / content_length if content_length > 0 else 0
            
            for field, (start_pos, end_pos) in field_ranges.items():
                start_idx = int(start_pos * logprobs_per_char)
                end_idx = int(end_pos * logprobs_per_char)
                end_idx = min(end_idx, len(all_logprobs))
                
                if start_idx < len(all_logprobs) and start_idx < end_idx:
                    field_logprobs[field] = all_logprobs[start_idx:end_idx]
        
        except Exception as e:
            print(f"Errore nell'analisi JSON: {e}")
            # Fallback: distribuzione uniforme
            return self._distribute_logprobs_uniformly(all_logprobs)
        
        #print("CAMPI\n",field_logprobs)
        return field_logprobs
    
    def _find_field_ranges(self, content: str) -> Dict[str, tuple]:
        """Trova i range di caratteri per ogni campo"""
        field_ranges = {}
        
        for field in self.expected_fields:
            # Cerca il pattern del campo
            pattern = f'"{field}"\\s*:\\s*"([^"]*)"'
            match = re.search(pattern, content)
            
            if match:
                start_pos = match.start(1)  # Inizio del valore
                end_pos = match.end(1)      # Fine del valore
                field_ranges[field] = (start_pos, end_pos)
        
        return field_ranges
    
    def _distribute_logprobs_uniformly(self, all_logprobs: List[float]) -> Dict[str, List[float]]:
        """Distribuzione uniforme come fallback"""
        field_logprobs = {field: [] for field in self.expected_fields}
        
        if not all_logprobs:
            return field_logprobs
        
        logprobs_per_field = len(all_logprobs) // len(self.expected_fields)
        remainder = len(all_logprobs) % len(self.expected_fields)
        
        start_idx = 0
        for i, field in enumerate(self.expected_fields):
            end_idx = start_idx + logprobs_per_field
            if i < remainder:
                end_idx += 1
            
            field_logprobs[field] = all_logprobs[start_idx:end_idx]
            start_idx = end_idx
        
        return field_logprobs
    
#
# Confidence Calculator
#
class ConfidenceCalculator:
    @staticmethod
    def calculate_field_confidence(logprobs: List[float]) -> float:
        """Calcola la confidence per un singolo campo"""
        if not logprobs:
            return 0.0
        try:
            # Converte log probabilities in probabilities e fa la media
            return float(np.exp(np.mean(logprobs)))
        except Exception as e:
            print(f"Errore nel calcolo confidence campo: {e}")
            return 0.0
    
    @staticmethod
    def calculate_overall_confidence(all_logprobs: List[float]) -> float:
        """Calcola la confidence complessiva"""
        if not all_logprobs:
            return 0.0
        try:
            return float(np.exp(np.mean(all_logprobs)))
        except Exception as e:
            print(f"Errore nel calcolo confidence complessiva: {e}")
            return 0.0
    
    @staticmethod
    def calculate_weighted_confidence(field_logprobs: Dict[str, List[float]]) -> float:
        """Calcola confidence pesata basata sulla lunghezza dei campi"""
        if not field_logprobs:
            return 0.0
        
        try:
            total_tokens = sum(len(logprobs) for logprobs in field_logprobs.values())
            if total_tokens == 0:
                return 0.0
            
            weighted_sum = 0.0
            for field, logprobs in field_logprobs.items():
                if logprobs:
                    field_confidence = np.exp(np.mean(logprobs))
                    weight = len(logprobs) / total_tokens
                    weighted_sum += field_confidence * weight
            
            return float(weighted_sum)
        except Exception as e:
            print(f"Errore nel calcolo confidence pesata: {e}")
            return 0.0
    
    @staticmethod
    def calculate_all_confidences(field_logprobs: Dict[str, List[float]], 
                                all_logprobs: List[float]) -> Dict[str, float]:
        """Calcola tutte le confidence"""
        result = {}
        
        # Confidence per campo specifico
        for field, logprobs in field_logprobs.items():
            result[f"{field}_confidence"] = ConfidenceCalculator.calculate_field_confidence(logprobs)
        
        # Confidence complessiva
        result["confidence"] = ConfidenceCalculator.calculate_overall_confidence(all_logprobs)
        
        # Confidence pesata (alternativa alla overall)
        #result["weighted_confidence"] = ConfidenceCalculator.calculate_weighted_confidence(field_logprobs)
        
        # Statistiche aggiuntive
        #result["total_tokens"] = len(all_logprobs)
        #result["fields_with_logprobs"] = len([f for f, lp in field_logprobs.items() if lp])
        
        return result


#
# Unified Stream Processor
#
class UnifiedStreamProcessor:
    def __init__(self, adapter: UnifiedAdapter, expected_fields: List[str], 
                 logprobs_service: Optional[LogprobsService] = None):
        self.adapter = adapter
        self.parser = JSONStreamParser(expected_fields)
        self.analyzer = JSONAnalyzer(expected_fields)
        self.confidence_calc = ConfidenceCalculator()
        self.logprobs_service = logprobs_service
        
        # Per accumulare contenuto quando usiamo logprobs service
        self.accumulated_content = ""
        self.system_prompt = ""
        self.user_prompt = ""
    
    def set_prompts(self, system_prompt: str, user_prompt: str):
        """Imposta i prompt per il calcolo logprobs"""
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.accumulated_content = ""
        self.parser.reset()  # Reset del parser per nuova sessione

    def process_streaming_response(self, response_stream, **kwargs):
        """Processa risposta in streaming"""
        needs_logprobs_calculation = self.logprobs_service is not None
        has_native_logprobs = False

        if not self.adapter.supports_native_usage_streaming():
            import math
            in_tokens = math.ceil((len(self.system_prompt) + len(self.user_prompt))/4.5)
            out_tokens = 0

        for event in response_stream:
            content = self.adapter.extract_content_streaming(event)
            logprobs = self.adapter.extract_logprobs_streaming(event)
            usage = self.adapter.extract_usage_streaming(event)

            if not self.adapter.supports_native_usage_streaming():
                out_tokens +=1

            if content:
                # CORREZIONE: Accumula SEMPRE il contenuto se ci sono logprobs
                if needs_logprobs_calculation or logprobs:  # ✅ AGGIUNTO "or logprobs"
                    self.accumulated_content += content
                
                # Se abbiamo logprobs nativi, usali
                if logprobs:
                    has_native_logprobs = True
                    parse_info = self.parser.process_chunk(content, logprobs)
                else:
                    # Altrimenti processa senza logprobs per ora
                    parse_info = self.parser.process_chunk(content, [])
                
                yield f"data: {json.dumps({'message': content})}\n\n"
            
            if usage or self.adapter.is_done_streaming(event):
                # Se dobbiamo calcolare logprobs e non li abbiamo nativi
                if (needs_logprobs_calculation and 
                    not has_native_logprobs and 
                    self.accumulated_content.strip()):
                    
                    print("Calculating logprobs using fallback service...")
                    calculated_logprobs = self.logprobs_service.calculate_streaming_logprobs(
                        self.system_prompt, 
                        self.user_prompt, 
                        self.accumulated_content
                    )
                    
                    # MODIFICA: Controlla se usare confidence pre-calcolate
                    if (hasattr(calculated_logprobs, 'metadata') and 
                        calculated_logprobs.metadata.get('use_precalculated_confidences', False)):
                        
                        # Usa le confidence già calcolate dal LocalModel
                        confidences = calculated_logprobs.metadata.get('field_confidences', {})
                        
                        # Assicurati che ci sia la confidence complessiva
                        if 'confidence' not in confidences:
                            confidences['confidence'] = calculated_logprobs.confidence_score
                        
                        print("DEBUG - Using pre-calculated confidences from LocalModel service")
                        print(f"DEBUG - Pre-calculated confidences: {confidences}")
                        
                    else:
                        # Comportamento normale: analizza e calcola
                        field_logprobs = self.analyzer.analyze_json_content(
                            self.accumulated_content, 
                            calculated_logprobs.logprobs
                        )
                        
                        confidences = self.confidence_calc.calculate_all_confidences(
                            field_logprobs,
                            calculated_logprobs.logprobs
                        )
                        
                else:
                    # CORREZIONE: Usa logprobs nativi con contenuto accumulato
                    print("Using native logprobs with field separation...")
                    
                    # DEBUG: Verifica cosa abbiamo
                    print(f"DEBUG - Accumulated content length: {len(self.accumulated_content)}")
                    print(f"DEBUG - Parser all_logprobs length: {len(self.parser.all_logprobs)}")
                    print(f"DEBUG - Content sample: '{self.accumulated_content[:100]}...'")
                    
                    if not self.accumulated_content.strip():
                        print("ERROR - No accumulated content for native logprobs!")
                        confidences = {'a_confidence': 0.0, 'r_confidence': 0.0, 'rt_confidence': 0.0, 'confidence': 0.0}
                    elif not self.parser.all_logprobs:
                        print("ERROR - No logprobs accumulated!")
                        confidences = {'a_confidence': 0.0, 'r_confidence': 0.0, 'rt_confidence': 0.0, 'confidence': 0.0}
                    else:
                        # Usa JSONAnalyzer per separare i logprobs nativi per campo
                        field_logprobs = self.analyzer.analyze_json_content(
                            self.accumulated_content, 
                            self.parser.all_logprobs
                        )
                        
                        print(f"DEBUG - Analyzer result: {[(k, len(v)) for k, v in field_logprobs.items()]}")
                        
                        # Se JSONAnalyzer fallisce, usa distribuzione semplice
                        if all(len(logprobs) == 0 for logprobs in field_logprobs.values()):
                            print("DEBUG - JSONAnalyzer failed, using simple distribution...")
                            
                            # Distribuzione semplice basata su lunghezze tipiche
                            total_logprobs = len(self.parser.all_logprobs)
                            if total_logprobs >= 3:
                                a_portion = int(total_logprobs * 0.7)  # 70% per analysis
                                r_portion = int(total_logprobs * 0.25) # 25% per response  
                                rt_portion = total_logprobs - a_portion - r_portion
                                
                                field_logprobs = {
                                    'a': self.parser.all_logprobs[0:a_portion],
                                    'r': self.parser.all_logprobs[a_portion:a_portion + r_portion],
                                    'rt': self.parser.all_logprobs[a_portion + r_portion:]
                                }
                                
                                print(f"DEBUG - Simple distribution: a={len(field_logprobs['a'])}, r={len(field_logprobs['r'])}, rt={len(field_logprobs['rt'])}")
                            else:
                                field_logprobs = {'a': [], 'r': [], 'rt': []}
                        
                        confidences = self.confidence_calc.calculate_all_confidences(
                            field_logprobs,
                            self.parser.all_logprobs
                        )
                        
                        print(f"DEBUG - Calculated confidences: {confidences}")
                
                if not self.adapter.supports_native_usage_streaming():
                    usage['prompt_tokens'] = in_tokens
                    usage['completion_tokens'] = out_tokens
                    usage['total_tokens'] = in_tokens + out_tokens

                result = {
                    'input': usage.get('prompt_tokens', 0) if usage else 0,
                    'output': usage.get('completion_tokens', 0) if usage else 0,
                    'total': usage.get('total_tokens', 0) if usage else 0,
                    **confidences,
                }
                
                print(f"Final confidences: {confidences}")
                yield f"data: {json.dumps(result)}\n\n"
                break

    def process_non_streaming_response(self, response) -> Dict:
        """Processa risposta non in streaming (senza logprobs)"""
        content = self.adapter.extract_content_non_streaming(response)
        usage = self.adapter.extract_usage_non_streaming(response)
        
        # Non calcoliamo logprobs per chiamate non-streaming
        confidences = {
            'a_confidence': 0.0,
            'r_confidence': 0.0,
            'rt_confidence': 0.0,
            'confidence': 0.0,
            'weighted_confidence': 0.0,
            'total_tokens': 0,
            'fields_with_logprobs': 0
        }
        
        
        return {
            'content': content,
            'usage': usage,
            'confidences': confidences,
            'field_logprobs': {},
            'has_native_logprobs': False,
            'used_fallback_logprobs': False
        }