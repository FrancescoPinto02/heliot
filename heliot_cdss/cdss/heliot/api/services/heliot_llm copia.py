from ...db_management import *
from ...patient_management import *
from ...ingredients_onthology import *
from ...dss_prompts import *
import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor #, as_completed
#import asyncio
import traceback
import json
import numpy as np
import yaml

#from typing import AsyncGenerator

class HeliotLLM:
    def __init__(self, db_uri:str ="drugs_db", synonym_csv:str ="ingredients_synonyms.csv", pt_db_uri:str ="medical_narrative"):
        self.dbm = DatabaseManagement(db_uri=db_uri, store_lower_case=True)
        self.ont = SynonymManager(synonym_csv)
        self.ptm = MedicalNarrativeDB(db_uri=pt_db_uri)

        self.model = "gpt-4o"
        self.url = None
        self._config()

        if self.url:
            self.client = OpenAI(base_url=self.url)
        else:
            # Initialize the OPENAI API
            OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
            self.client = OpenAI(
                # This is the default and can be omitted
                api_key=os.environ.get("OPENAI_API_KEY"),
            )

    def _config(self):
        with open("config.yml", 'r') as file:
            config_data = yaml.safe_load(file)
        conf = config_data.get('config', [])
        for c in conf:
            self.model = c.get('model',"gpt4-o")
            self.url = c.get('url',"")

    def _extract_composition_from_clinical_notes(self, text:str) -> str:
        try:
            response = self.client.chat.completions.create(model=self.model,
                                    messages=[{"role": "system", "content": ""},
                                            {"role": "user", "content":  USER_EXTRACT_COMPOSITION.format(narrative=text)}],
                                    max_tokens=3000,
                                    temperature = 0)
            cleaned_text = response.choices[0].message.content
            return cleaned_text
        except Exception as e:
            print(f"Failed to process text with the LLM: {e}")
            return None

    # Translate the text in English
    def _translate_in_english(self, text:str) -> str:
        try:
            response = self.client.chat.completions.create(model=self.model,
                                    messages=[{"role": "system", "content": ""},
                                            {"role": "user", "content":  USER_ENGLISH_TRANSLATION.format(text=text)}],
                                    max_tokens=3000,
                                    temperature = 0)
            #print(USER_ENGLISH_TRANSLATION.format(text=text))
            cleaned_text = response.choices[0].message.content
            return cleaned_text
        except Exception as e:
            print(f"Failed to process translation with the LLM: {e}")
            return None

    def _internal_search_drug(self,drug_code:str)-> Dict:
        #print("SEARCHING...", drug_code)
        return self.dbm.search_drug(drug_code)

    def _internal_search_patient(self, patient_id:str)-> Dict:
        #print("SEARCHING...", patient_id)
        return self.ptm.search_patient(patient_id)
    

    def _chat_completion_create(self, model, messages, max_tokens, temperature, stream):
        stream = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream
        )
        for event in stream:
                if event.choices[0].delta.content is not None:
                    #print(event.choices[0].delta.content, end="")
                    yield f"data: {json.dumps({'message': event.choices[0].delta.content})}\n\n"

    def dss_check(self, drug_code: str, allergy: str):
        #print("DRUG CODE", drug_code)
        
        if len(allergy) >0:
            allergy = allergy.lower()
            with ThreadPoolExecutor() as executor:
                future_drug = executor.submit(self._internal_search_drug, drug_code)
                future_translation = executor.submit(self._translate_in_english, allergy)
                    
                # Wait for the results
                drg = future_drug.result()
                allergy = future_translation.result()
                al = self.ont.find_standard_name(allergy)
                allergy_type = "allergic to "+al
        else:
            allergy_type = "not allergic"
            drg = self.dbm.search_drug(drug_code)

        composition = drg['composition']
        excipients = drg['excipients']
        prescription = drg['drug_name']

        try:
            response = self.client.chat.completions.create(model=self.model,
                                    messages=[{"role": "system", "content": SYSTEM_CHECK_ALLERGY_PROMPT.format(drug=prescription, active_ingredients=composition, excipients=excipients)},
                                            {"role": "user", "content":  USER_CHECK_ALLERGY_PROMPT.format( allergy=allergy_type)}],
                                    max_tokens=3000,
                                    temperature = 0,
                                    stream=True,
                                    stream_options= {"include_usage": True})
            for event in response:
                if event.choices is not None and len(event.choices)>0 and event.choices[0].delta.content is not None:
                    yield f"data: {json.dumps({'message': event.choices[0].delta.content})}\n\n"
                if hasattr(event, 'usage') and event.usage is not None:
                    print(event.usage)
                    prompt_tokens = event.usage.prompt_tokens
                    completion_tokens = event.usage.completion_tokens
                    total_tokens = event.usage.total_tokens
                    yield f"data: {json.dumps({'input': prompt_tokens, 'output': completion_tokens, 'total':  total_tokens})}\n\n"
        except Exception as e:
            stack_trace = traceback.format_exc()
            
            # There is an exception
            print("Exception:")
            print(stack_trace)
            yield None

    def parallel_translate(self,comps)->list:
        with ThreadPoolExecutor() as executor:
            # Map: return the results in the same input order
            translated_comps = list(executor.map(self._translate_in_english, comps))
        return translated_comps

    def dss_check_enhanced(self, patient_id: str, drug_code: str, clinical_notes: str, store: bool = False):
        #print("DRUG CODE", drug_code)
        
        # If there are clinical_notes
        if len(clinical_notes.strip()) >0:
            clinical_notes = clinical_notes.lower()
            with ThreadPoolExecutor() as executor:
                future_drug = executor.submit(self._internal_search_drug, drug_code)
                future_patient = executor.submit(self._internal_search_patient, patient_id)
                future_translation = executor.submit(self._extract_composition_from_clinical_notes, clinical_notes)
                    
                # Wait for the results
                drg = future_drug.result()
                comps = future_translation.result()

                # Replace synonyms in text
                if comps and len(comps.strip())>0:
                    # preserve the original names
                    orig_comps = comps.split("#")
                    orig_comps = [x.strip() for x in orig_comps if x.strip()]
                    if orig_comps:
                        # translate them in English
                        comps = self.parallel_translate(orig_comps)
                        s_ingredients = self.ont.find_standard_names(comps)
                        print(orig_comps, "\n", comps, "\n", s_ingredients)
                        for i in range(len(s_ingredients)):
                            clinical_notes = clinical_notes.replace(orig_comps[i],s_ingredients[i].get('t'))  
                    print("NEW CLINICAL NOTES", clinical_notes)

                patient_info = clinical_notes
                pt = future_patient.result()
                print("PATIENT", pt)
                if pt:
                    patient_info += "\n"+pt['clinical_notes']

        else:
            patient_info = "not allergic"
            with ThreadPoolExecutor() as executor:
                future_drug = executor.submit(self._internal_search_drug, drug_code)
                future_patient = executor.submit(self._internal_search_patient, patient_id)
                drg = future_drug.result()
                pt = self._internal_search_patient(patient_id)
                print("PATIENT", pt)
                if pt:
                    patient_info = pt['clinical_notes']

        composition = drg['composition']
        excipients = drg['excipients']

        # TRANSL HERE
        comps = self.parallel_translate(composition)
        ex_comps = self.parallel_translate(excipients)
        s_act = self.ont.find_standard_names(comps)
        s_ex = self.ont.find_standard_names(ex_comps)
        for i in range(len(s_act)):
            composition[i] = s_act[i].get('t').replace("\\n","").replace("\n","")
        for i in range(len(s_ex)):
            excipients[i] = s_ex[i].get('t').replace("\\n","").replace("\n","")

        prescription = drg['drug_name']
        contraindications = drg['contraindications']
        cross_reactivity = drg['cross_reactivity']
        if cross_reactivity.find("{'description': '', 'incidence': '', 'da': '', 'cross_sensitive_drugs': []}") !=-1:
            cross_reactivity = ""

        # Provide the final answer  
        try:
            if self.url:
                sy_inst = OW_SYSTEM_CHECK_ALLERGY_ENHANCED_PROMPT
            else:
                sy_inst = SYSTEM_CHECK_ALLERGY_ENHANCED_PROMPT
            response = self.client.chat.completions.create(model=self.model,
                                    messages=[{"role": "system", "content": sy_inst.format(drug=prescription, active_ingredients=composition, excipients=excipients, cross_reactivity=cross_reactivity, contraindications=contraindications)},
                                            {"role": "user", "content":  USER_CHECK_ALLERGY_ENHANCED_PROMPT.format( patient_info=patient_info)}],
                                    max_tokens=3000,
                                    temperature = 0,
                                    stream=True,
                                    logprobs=True,
                                    top_logprobs=1,
                                    stream_options= {"include_usage": True})
            
            print(sy_inst.format(drug=prescription, active_ingredients=composition, excipients=excipients, cross_reactivity=cross_reactivity, contraindications=contraindications))
            print("\n")
            print(USER_CHECK_ALLERGY_ENHANCED_PROMPT.format( patient_info=patient_info))
            token_logprobs = []
            analysis_token_logprobs = []
            case_token_logprobs = []
            reactionType_logprobs = []

            is_in_field = False #":\""
            field = -1 # -1 = None, 0 = a, 1 = r, 2 = rt
            collect = False
            is_end_message = False #\"}"

            no_usage = True
            for event in response:

                if event.choices is not None and len(event.choices)>0 and event.choices[0].delta.content is not None:
                    content = event.choices[0].delta.content
                    if "\"}" in content:
                        is_end_message = True
                        collect = False
                    if not is_end_message:
                        if "{\"" in content or ",\"" in content:
                            is_in_field = True
                            collect = False
                        if is_in_field and (content == "a" or content=="r" or content == "rt"):
                            collect = False
                            field += 1

                    #print(event)

                    if event.choices[0].logprobs and event.choices[0].logprobs.content:

                        for logs in event.choices[0].logprobs.content:
                            token_logprobs.append(logs.logprob)
                            if collect:
                                if field == 0:
                                    analysis_token_logprobs.append(logs.logprob)
                                elif field == 1:
                                    case_token_logprobs.append(logs.logprob)
                                else:
                                    reactionType_logprobs.append(logs.logprob)
                    if ":\"" in content:
                        collect = True

                    yield f"data: {json.dumps({'message': content})}\n\n"
                if hasattr(event, 'usage') and event.usage is not None:
                    no_usage = False
                    #print(event.usage)
                    prompt_tokens = event.usage.prompt_tokens
                    completion_tokens = event.usage.completion_tokens
                    total_tokens = event.usage.total_tokens
                    try:
                        confidence = np.exp(np.mean(token_logprobs))
                        analysis_confidence = np.exp(np.mean(analysis_token_logprobs))
                        case_confidence = np.exp(np.mean(case_token_logprobs))
                        reaction_confidence = np.exp(np.mean(reactionType_logprobs))
                    except Exception as ec:
                        confidence = 0.0
                        analysis_confidence = 0.0
                        case_confidence = 0.0
                        reaction_confidence = 0.0
                    print(f"'confidence': {confidence}, 'analysis_confidence':{analysis_confidence}, 'case_confidence':{case_confidence}, 'reaction_confidence':{reaction_confidence}")
                    yield f"data: {json.dumps({'input': prompt_tokens, 'output': completion_tokens, 'total':  total_tokens, 'confidence': confidence, 'analysis_confidence':analysis_confidence, 'case_confidence':case_confidence, 'reaction_confidence':reaction_confidence})}\n\n"
                    #yield f"data: {json.dumps({'input': prompt_tokens, 'output': completion_tokens, 'total':  total_tokens})}\n\n"
    
            if no_usage:
                print("NOW===>")
                yield f"data: {json.dumps({'input': 0, 'output': 0, 'total':  0, 'confidence': 0.0, 'analysis_confidence':0.0, 'case_confidence':0.0, 'reaction_confidence':0.0})}\n\n"
            if store and clinical_notes:
                self.ptm.update_patient(patient_id,clinical_notes)
                print(SYSTEM_CHECK_ALLERGY_ENHANCED_PROMPT.format(drug=prescription, active_ingredients=composition, excipients=excipients, cross_reactivity=cross_reactivity, contraindications=contraindications))
                print("\n",USER_CHECK_ALLERGY_ENHANCED_PROMPT.format( patient_info=patient_info))
        except Exception as e:
            stack_trace = traceback.format_exc()
            
            # There is an exception
            print("Exception:")
            print(stack_trace)
            yield None