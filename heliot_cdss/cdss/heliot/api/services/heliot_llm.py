from ...db_management import *
from ...patient_management import *
from ...ingredients_onthology import *
from ...dss_prompts import *
from .llm_adapter import *
from .cache import *

import importlib
import sys
import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
import traceback
import json
import numpy as np
import yaml
import re
import traceback

from typing import Any, Dict, Optional

import threading

#
# Heliot Controller
#
class HeliotLLM:
    def __init__(self, synonym_csv:str ="ingredients_synonyms.csv", pt_db_uri:str ="medical_narrative"):

        self.ont = SynonymManager(synonym_csv)
        self.ptm = MedicalNarrativeDB(db_uri=pt_db_uri)

        self.model = "gpt-4o"
        self.url = None
        self.family = "openai"
        self.client = None
        self.temperature = 0
        self.MAX_TOKENS = 3000

        self.cache = ThreadSafeCache(default_ttl=200)
        self._config()
        self._setup_adapter()
        self._setup_drug_management()

    #
    # Setup the drug management class
    #
    def _setup_drug_management(self):
        print(self.drug_management_config)
        drug_mgt_module_path = self.drug_management_config.get('module_name','cdss.db_management')
        mgt_class = self.drug_management_config.get('class_name','DatabaseManagement')
        mgt_config = self.drug_management_config.get('config',{'db_uri':"drugs_db", 'store_lower_case':True})
        self.dbm = self.load_drug_manager_generic(drug_mgt_module_path, mgt_class, mgt_config) #DatabaseManagement(db_uri=db_uri, store_lower_case=True)

    #
    # Load the drug manager class dynamically
    #
    def load_drug_manager_generic(self, module_path: str, class_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Generic method to load any drug management class.
        
        Args:
            module_path (str): The module path (e.g., 'cdss.db_management' or '...db_management' for relative)
            class_name (str): The class name (e.g., 'DatabaseManagement')
            config (Optional[Dict[str, Any]]): Configuration for the database manager
        """
        try:
            # Se il module_path inizia con dei punti, usa import relativo
            if module_path.startswith('.'):
                # Ottieni il package corrente
                current_package = __name__.rsplit('.', 1)[0] if '.' in __name__ else None
                module = importlib.import_module(module_path, package=current_package)
            else:
                module = importlib.import_module(module_path)
                
            database_class = getattr(module, class_name)
            
            if not issubclass(database_class, BaseDatabaseManagement):
                raise TypeError(f"{class_name} is not a subclass of BaseDatabaseManagement")
            
            db_manager = database_class(config)
            print(f"Successfully loaded: {database_class.__name__} from {module_path}")
            return db_manager
            
        except ImportError as e:
            print(f"Error importing module '{module_path}': {e}")
            raise
        except AttributeError as e:
            print(f"Class '{class_name}' not found in module '{module_path}': {e}")
            raise

    #
    # Use the yaml configuration to configure HELIOT controller
    #
    def _config(self):
        with open("config.yml", 'r') as file:
            config_data = yaml.safe_load(file)
            conf = config_data.get('config', [])
            for c in conf:
                self.model = c.get('model', "gpt-4o")
                self.url = c.get('url', "")
                self.family = c.get('family', 'openai')
                self.temperature = c.get('temperature', 0)
                self.MAX_TOKENS = c.get('max_tokens', 3000)
                self.logprobs_service = c.get('logprobs_service',{})
                self.drug_management_config = c.get('drug_management',{})
                self.tmodel = c.get('tmodel', self.model)
                self.tfamily = c.get('tfamily', self.family)
                self.turl = c.get('turl', self.url)
                self.cot_prompt = c.get('cot_prompt', False)
                self.safe_patterns = c.get('safe_patterns','')
                self.need_proc_all = c.get('need_proc_all', False)
        self._setLLMClient()

    #
    # Set the LLM client to use
    #
    def _setLLMClient(self):
        if self.family == "openai":
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), timeout=30)
        elif self.family == "openllm":  # Nuovo family type
            self.client = OpenAI(base_url=self.url)
        elif self.family == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        elif self.family == "gemini":
            from google import genai
            self.client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        else:
            raise ValueError(f"Invalid configuration option for LLM model family: {self.family}")
        
        if self.family == self.tfamily:
            self.tclient = self.client
        else:
            if self.tfamily == "openai":
                self.tclient = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),timeout=5)
            elif self.tfamily == "openllm":  # Nuovo family type
                self.tclient = OpenAI(base_url=self.turl, timeout=5)
            elif self.tfamily == "anthropic":
                import anthropic
                self.tclient = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
            elif self.tfamily == "gemini":
                from google import genai
                self.tclient = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
            else:
                raise ValueError(f"Invalid configuration option for LLM model family: {self.family}")            
    
    #
    # Setup the most appropriate adapter for the LLM
    #
    def _setup_adapter(self):

        if self.family == "openai":
            self.adapter = OpenAIUnifiedAdapter()
        elif self.family == "openllm":
            self.adapter = OpenLLMUnifiedAdapter()
        elif self.family == "anthropic":
            self.adapter = AnthropicUnifiedAdapter()
        elif self.family == "gemini":
            self.adapter = GeminiUnifiedAdapter()


        if self.tfamily == "openai":
            self.tadapter = OpenAIUnifiedAdapter()
        elif self.tfamily == "openllm":
            self.tadapter = OpenLLMUnifiedAdapter()
        elif self.tfamily == "anthropic":
            self.adapter = AnthropicUnifiedAdapter()
        elif self.tfamily == "gemini":
            self.tadapter = GeminiUnifiedAdapter()

        # Determine for the selected LLM we need a logprobs service
        logprobs_service = None
        
        needs_fallback_logprobs = (
            self.logprobs_service and 
            self.family not in ('openai')  # Only OpenAI has native logprobs
        )
        
        print(self.logprobs_service, needs_fallback_logprobs)
        print(self.logprobs_service.get('name',''))
        if needs_fallback_logprobs:
            from .base_prob_serv import LogprobsServiceType
            
            if self.logprobs_service.get('name','') == "local":
                from .local_prob_serv import LocalModelLogprobsService
                logprobs_service = LocalModelLogprobsService(
                    self.logprobs_service
                )
    
            print(f"Setup logprobs fallback service for {self.family} using {self.logprobs_service}")
        
        # Create the unified processor with or without a logprobs service
        self.processor = UnifiedStreamProcessor(
            self.adapter, 
            ["a", "r", "rt"], 
            logprobs_service
        )


        self.tprocessor = UnifiedStreamProcessor(
            self.tadapter, 
            ["a", "r", "rt"], 
            logprobs_service
        )

    #
    # Perform the LLM call using the right client
    #
    def _callLLM(self, system_prompt, user_prompt, stream=False):
        """Chiamata LLM unificata"""
        if self.family in ('openai', 'openllm'):
            params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": self.MAX_TOKENS,
                "temperature": self.temperature,
                "timeout": 30,
            }
            
            if stream:
                params["stream"] = True
                params["stream_options"] = {"include_usage": True}
                # Only OpenAI supports logprobs as parameter
                if self.family == "openai":
                    params["logprobs"] = True
                    params["top_logprobs"] = 5
            
            response = self.client.chat.completions.create(**params)
            
        elif self.family == "anthropic":
            msg = [{"role": "user", "content": user_prompt}]
            params = {
                "model": self.model,
                "max_tokens": self.MAX_TOKENS,
                "system": system_prompt,
                "messages": msg,
                "temperature": self.temperature,
            }
            
            if stream:
                params["stream"] = True
            
            response = self.client.messages.create(**params)
            
        elif self.family == "gemini":
            from google.genai import types 
            params = {
                "model": self.model,
                "contents": types.Part.from_text(user_prompt),
                "config": types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=self.temperature,
                    maxOutputTokens=self.MAX_TOKENS
                ),
            }
            
            if stream:
                params["stream"] = True
            
            response = self.client.models.generate_content(**params)
        
        return response


    #
    # Perform the LLM call using the right client
    #
    def _callLLMForTranslation(self, system_prompt, user_prompt, stream=False):
        """Chiamata LLM unificata"""
        if self.tfamily in ('openai', 'openllm'):
            params = {
                "model": self.tmodel,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": self.MAX_TOKENS,
                "temperature": self.temperature,
                "timeout": 5,
            }
            
            if stream:
                params["stream"] = True
                params["stream_options"] = {"include_usage": True}
                # Only OpenAI supports logprobs as parameter
                if self.tfamily == "openai":
                    params["logprobs"] = True
                    params["top_logprobs"] = 5
            try:
                print('Eseguo la chiamata')
                response = self.tclient.chat.completions.create(**params)
            except Exception as ex:
                print(traceback.format_exc())
                print('TIMEOUT')
                time.sleep(1)
                response = self.tclient.chat.completions.create(**params)
                print('Tutto ok qui=============>')
            
        elif self.tfamily == "anthropic":
            msg = [{"role": "user", "content": user_prompt}]
            params = {
                "model": self.tmodel,
                "max_tokens": self.MAX_TOKENS,
                "system": system_prompt,
                "messages": msg,
                "temperature": self.temperature,
            }
            
            if stream:
                params["stream"] = True
            
            response = self.tclient.messages.create(**params)
            
        elif self.tfamily == "gemini":
            from google.genai import types 
            params = {
                "model": self.tmodel,
                "contents": types.Part.from_text(user_prompt),
                "config": types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=self.temperature,
                    maxOutputTokens=self.MAX_TOKENS
                ),
            }
            
            if stream:
                params["stream"] = True
            
            try:
                response = self.tclient.models.generate_content(**params)
            except Exception as ex:
                print('Riprovo')
                time.sleep(1)
                response = self.tclient.models.generate_content(**params)
        
        return response
    
    #
    # Process the LLM response
    #
    def process_response(self, system_prompt, user_prompt, stream=False, **kwargs):
        """Processa la risposta in modo unificato"""
        response = self._callLLM(system_prompt, user_prompt, stream=stream)
        
        if stream:
            # set prompts 
            self.processor.set_prompts(system_prompt, user_prompt)
            return self.processor.process_streaming_response(response, **kwargs)
        else:
            return self.processor.process_non_streaming_response(response)


    #
    # Process the LLM response
    #
    def process_response_forTranslation(self, system_prompt, user_prompt, stream=False, **kwargs):
        """Processa la risposta in modo unificato"""
        response = self._callLLMForTranslation(system_prompt, user_prompt, stream=stream)
        return self.tprocessor.process_non_streaming_response(response)

    #
    # Extract compounds, drugs, or substances from the clinical notes.
    #
    def _extract_composition_from_clinical_notes(self, text:str) -> str:
        print('Entrato in _extract_composition_from_clinical_notes')
        res = self.cache.get(text)
        if res is not None:
            print('Found in cache')
            return res
        try:        
            #print('_extract_composition_from_clinical_notes')
            result = self.process_response_forTranslation(
                system_prompt="",  
                user_prompt=USER_EXTRACT_COMPOSITION.format(narrative=text),
                stream=False  
            )
            res = result.get('content', '')
            #print('_res_extract_from_clinical_notes', res)
            if res.upper().find('NO_NO') !=-1:
                res = ''
            self.cache.put(text, res)
            return res
        except Exception as e:
            print(f"Failed to process text with the LLM: {e}")
            return None

    #
    # Translate the text in English
    #
    def _translate_in_english(self, text: str) -> str:
        res = self.cache.get(('trns',text))
        if res is not None:
            print('Found in cache')
            return res
        try:
            #print("\n===>")
            #print(USER_ENGLISH_TRANSLATION.format(text=text))
            result = self.process_response_forTranslation(
                system_prompt="",  
                user_prompt=USER_ENGLISH_TRANSLATION.format(text=text),
                stream=False  
            )
            
            cleaned_text = result.get('content', '')
            #print("ANSWER:", cleaned_text)
            # Optional Debug
            if hasattr(self, 'debug_translation') and self.debug_translation:
                print(f"Translation input: {text}")
                print(f"Translation output: {cleaned_text}")
                print(f"Usage: {result.get('usage', {})}")
                print(f"Logprobs source: {result.get('logprobs_source', 'none')}")
            
            #print(f'_translated content: {text} -> {result}')

            self.cache.put(('trns',text),cleaned_text)
            return cleaned_text
            
        except Exception as e:
            print(f"Failed to process translation with the LLM: {e}")
            return None

    #
    # Search the drug info
    #
    def _internal_search_drug(self,drug_code:str)-> Dict:
        print("SEARCHING...", drug_code)
        res = self.cache.get(('drg',drug_code))
        if res is not None:
            print('found in cache')
            return res
        res = self.dbm.search_drug(drug_code)
        self.cache.put(('drg',drug_code),res)
        return res

    #
    # Search the patient info
    #
    def _internal_search_patient(self, patient_id:str)-> Dict:
        print(f"[LOCK-DEBUG] Inizio _internal_search_patient, thread: {threading.current_thread().name}")
        print(f"[LOCK-DEBUG] Thread attivi: {threading.active_count()}")
        return self.ptm.search_patient(patient_id)
    
    #
    # Parallel translate a list of compounds
    #
    def parallel_translate(self,comps)->list:
        res = self.cache.get(comps)
        if res is not None:
            print('Found in cache')
            return res
        max_workers = min(32, (os.cpu_count() or 1) * 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Map: return the results in the same input order
            translated_comps = list(executor.map(self._translate_in_english, comps))
        self.cache.put(comps,translated_comps)
        return translated_comps

    def _processIngredientsInClinicalNotes(self, comps, clinical_notes):
        # preserve the original names
        orig_comps = comps.split("#")
        orig_comps = [x.strip() for x in orig_comps if x.strip()]
        print("BEFORE TRANSLATION", orig_comps)
        if orig_comps:
            # translate them in English
            comps = self.parallel_translate(orig_comps)
                        
            s_ingredients = self.ont.find_standard_names(comps)
            print(orig_comps, "\n", comps, "\n", s_ingredients)
            for i in range(len(s_ingredients)):
                clinical_notes = clinical_notes.replace(orig_comps[i], s_ingredients[i].get('t'))  
        return clinical_notes


    def process_compositions_and_excipients(self, compositions, excipients):
        max_workers = min(10, len(compositions) + len(excipients))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks for compositions and excipients
            print('Avviato process_compositions_and_excipients')
            print(compositions)
            print("\n")
            print(excipients)

            results_compositions = self.cache.get(compositions)
            results_excipients = self.cache.get(excipients)

            if results_compositions is None:
                print("composition not found.....")
                results_compositions = []
                future_compositions = [executor.submit(self._translate_in_english, comp) for comp in compositions]
            else:
                print('Found in cache results_compositions')

            if results_excipients is None:
                print("excipients not found.....")
                results_excipients = []
                future_excipients = [executor.submit(self._translate_in_english, exc) for exc in excipients]

            raise_exception = False
            # Collect results for compositions
            if not results_compositions:
                for future in future_compositions:
                    try:
                        result = future.result(timeout=30)
                        results_compositions.append(result)
                    except Exception as e:
                        print(traceback.format_exc())
                        raise_exception = True
                        print(f"Exception occurred: {str(e)}")
                        results_compositions.append(None)
                if not raise_exception:
                    self.cache.put(compositions, results_compositions)
                else:
                    results_compositions = compositions

            # Collect results for excipients
            if not results_excipients:
                for future in future_excipients:
                    try:
                        result = future.result(timeout=30)
                        results_excipients.append(result)
                    except Exception as e:
                        print(traceback.format_exc())
                        raise_exception = True
                        print(f"Exception occurred: {str(e)}")
                        results_excipients.append(None)
                if not raise_exception:
                    self.cache.put(excipients, results_excipients)
                else:
                    results_excipients = future_excipients
            else:
                print('Found in cache results_excipients')

            if raise_exception:
                if future_compositions:
                    for f in future_compositions:
                        f.cancel()
                if future_excipients:
                    for f in future_excipients:
                        f.cancel()
                executor.shutdown(wait=False)
                raise

            print("_end of process_comp_and_exc")
            print(results_compositions)
            print("\n")
            print(results_excipients)

            return results_compositions, results_excipients


    def _gather_base_info(self, patient_id, drug_code, clinical_notes):
        print(f"[DEBUG] Thread attivi prima di creare executor: {threading.active_count()}")
        print(f"[DEBUG] Thread attivi: {[t.name for t in threading.enumerate()]}")
        max_workers = 3 #min(32, (os.cpu_count() or 1) * 4)
        import gc
        gc.collect()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            print(f'[DEBUG] Creato executor con {max_workers} workers')
            print(f"[DEBUG] Thread attivi dopo creazione executor: {threading.active_count()}")
            print('search base info')

            futures = {
                    executor.submit(self._internal_search_drug,     drug_code):      'drg',
                    executor.submit(self._internal_search_patient,  patient_id):     'pt',
                    executor.submit(self._extract_composition_from_clinical_notes, clinical_notes): 'comps',
            }

            print(f"[DEBUG] Thread attivi dopo submit: {threading.active_count()}")
            # Wait for the results
            print('waiting base info')
            results = {}
            try:
                for future in as_completed(futures, timeout=30):
                    key = futures[future]
                    print(f"[DEBUG] Aspetto key {key}, running: {future.running()}, done: {future.done()}")
                    results[key] = future.result()                   
            except Exception as ex:
                for f in futures:
                    f.cancel()
                print("Timeout during gathering futures")
                executor.shutdown(wait=False)
                print(f"[DEBUG] Thread attivi alla fine in execpt: {threading.active_count()}")
                raise

            print(f"[DEBUG] Thread attivi alla fine: {threading.active_count()}")
            return results


    def _sse(self, payload: dict) -> str:
        """
        Format a Server-Sent Events (SSE) message line.
        Produces: data: <json>\n\n
        """
        return "data: " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n\n"


    #
    # Perform the DSS logic
    #
    def dss_check_enhanced(self, patient_id: str, drug_code: str, clinical_notes: str, store: bool = False):
        try:
            print('called....')
            
            results = self._gather_base_info(patient_id, drug_code, clinical_notes)

            print('Eseguito _gather_base_info')
            drg = results['drg']
            comps = results['comps']
            pt = results['pt']
            print("PATIENT", pt)

            print('clinical translations....')
            # If we have clinical notes in input then we translate the compounds and ingredients
            if len(clinical_notes.strip()) > 0:
                clinical_notes = clinical_notes.lower()                
                # Replace synonyms in text
                if comps and len(comps.strip()) > 0:
                    clinical_notes = self._processIngredientsInClinicalNotes(comps, clinical_notes)
                patient_info = clinical_notes
            else:
                patient_info = "not allergic"
            
            # Assuming that the clinical notes are more recent
            if self.safe_patterns:
                print(patient_info,'\n', self.safe_patterns)
                pattern = re.compile(self.safe_patterns.strip())
                match = pattern.search(patient_info)
                #match = re.search(self.safe_patterns, patient_info, re.IGNORECASE)
                print(match)
                if match:
                    print('SECURE ANSWER ===========>')

                    msg_obj = {
                        "a": "The patient has no known allergies or reactions documented in their information.",
                        "r": "NO DOCUMENTED REACTIONS OR INTOLERANCES",
                        "rt": "None",
                    }

                    usage_obj = {
                        "input": 2000,
                        "output": 85,
                        "total": 2085,
                        "confidence": 0.888105228,
                        "a_confidence": 0.746244826,
                        "r_confidence": 0.999996548,
                        "rt_confidence": 0.999999806,
                    }

                    yield self._sse({"message": msg_obj})
                    yield self._sse(usage_obj)
                    return
                
            if pt:
                patient_info += "\n" + pt['clinical_notes']

            composition = drg['composition']
            excipients = drg['excipients']

            if self.need_proc_all:
                print('compounds translations....')
                # Here we translate the drugs active ingredients and excipients, and map them to standard names
                comps, ex_comps = self.process_compositions_and_excipients(composition,excipients)

                s_act = self.ont.find_standard_names(comps)
                s_ex = self.ont.find_standard_names(ex_comps)
                
                for i in range(len(s_act)):
                    composition[i] = s_act[i].get('t').replace("\\n", "").replace("\n", "")
                for i in range(len(s_ex)):
                    excipients[i] = s_ex[i].get('t').replace("\\n", "").replace("\n", "")

            prescription = drg['drug_name']
            contraindications = drg['contraindications']
            cross_reactivity = drg['cross_reactivity']
            if cross_reactivity.find("{'description': '', 'incidence': '', 'da': '', 'cross_sensitive_drugs': []}") != -1:
                cross_reactivity = ""

        except Exception as e:
            print("Exception....", str(e))
            # Gestisci altri errori
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return
        
        # CDSS logig for the final answer  
        try:
            # Determine the right prompt to use
            if self.cot_prompt:  
                sy_inst = OW_SYSTEM_CHECK_ALLERGY_ENHANCED_PROMPT
            else:
                sy_inst = SYSTEM_CHECK_ALLERGY_ENHANCED_PROMPT
            
            # Create the prompt
            system_prompt = sy_inst.format(
                drug=prescription, 
                active_ingredients=composition, 
                excipients=excipients, 
                cross_reactivity=cross_reactivity, 
                contraindications=contraindications
            )
            
            user_prompt = USER_CHECK_ALLERGY_ENHANCED_PROMPT.format(patient_info=patient_info)
            
            print(system_prompt)
            print("\n")
            print(user_prompt)
            print("\n")
            print("PROCESSING RESPONSE =======>")
            # Use the unified processor to process the LLM answer
            for result in self.process_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                stream=True,
                patient_id=patient_id,
                clinical_notes=clinical_notes,
                store=store
            ):
                yield result
                
            # Store the new clinical notes for the patient
            if store and clinical_notes:
                self.ptm.update_patient(patient_id, clinical_notes)
                
        except Exception as e:
            stack_trace = traceback.format_exc()
            print("Exception:")
            print(stack_trace)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"