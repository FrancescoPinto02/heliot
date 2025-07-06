from typing import Any, Dict, List, Optional, Union
import requests
from dataclasses import dataclass
from .base_db_mgt import *

@dataclass
class DrugInfo:
    """
    Data class to hold drug information retrieved from FHIR server.
    """
    name: str
    active_ingredients: List[str]
    excipients: List[str]
    contraindications: List[str]
    cross_reactivity: List[str]

class FHIRDatabaseManagement(BaseDatabaseManagement):
    """
    Concrete implementation of BaseDatabaseManagement using FHIR API.
    
    This class implements drug searching functionality using FHIR (Fast Healthcare 
    Interoperability Resources) standard to retrieve drug information.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the FHIR database management instance.
        
        Args:
            config (Optional[Dict[str, Any]]): Configuration dictionary that should contain:
                - 'base_url': FHIR server base URL
                - 'timeout': Request timeout in seconds (default: 30)
                - 'headers': Additional HTTP headers (optional)
                - 'auth': Authentication credentials (optional)
        """
        super().__init__(config)
        self.base_url = self.config.get('base_url', 'https://hapi.fhir.org/baseR4')
        self.timeout = self.config.get('timeout', 30)
        self.headers = self.config.get('headers', {'Accept': 'application/fhir+json'})
        self.auth = self.config.get('auth', None)
    
    def search_drug(self, drug_code: str) -> Optional[DrugInfo]:
        """
        Search for a drug in the FHIR server using its code.
        
        Args:
            drug_code (str): The unique code identifier for the drug
            
        Returns:
            Optional[DrugInfo]: DrugInfo object containing drug details if found,
                              None if not found or error occurred
                              
        Raises:
            requests.RequestException: If there's an error with the HTTP request
            ValueError: If the drug_code is empty or invalid
        """
        if not drug_code or not drug_code.strip():
            raise ValueError("Drug code cannot be empty")
        
        try:
            # Search for Medication resource by code
            medication_data = self._search_medication(drug_code)
            if not medication_data:
                return None
            
            # Search for MedicationKnowledge resource for additional details
            knowledge_data = self._search_medication_knowledge(drug_code)
            
            return self._parse_drug_info(medication_data, knowledge_data)
            
        except requests.RequestException as e:
            print(f"Error occurred while searching for drug {drug_code}: {e}")
            raise
    
    def _search_medication(self, drug_code: str) -> Optional[Dict]:
        """
        Search for Medication resource using FHIR API.
        
        Args:
            drug_code (str): The drug code to search for
            
        Returns:
            Optional[Dict]: Medication resource data if found, None otherwise
        """
        url = f"{self.base_url}/Medication"
        params = {'code': drug_code, '_format': 'json'}
        
        response = requests.get(
            url, 
            params=params, 
            headers=self.headers, 
            timeout=self.timeout,
            auth=self.auth
        )
        response.raise_for_status()
        
        bundle = response.json()
        entries = bundle.get('entry', [])
        
        return entries[0]['resource'] if entries else None
    
    def _search_medication_knowledge(self, drug_code: str) -> Optional[Dict]:
        """
        Search for MedicationKnowledge resource using FHIR API.
        
        Args:
            drug_code (str): The drug code to search for
            
        Returns:
            Optional[Dict]: MedicationKnowledge resource data if found, None otherwise
        """
        url = f"{self.base_url}/MedicationKnowledge"
        params = {'code': drug_code, '_format': 'json'}
        
        try:
            response = requests.get(
                url, 
                params=params, 
                headers=self.headers, 
                timeout=self.timeout,
                auth=self.auth
            )
            response.raise_for_status()
            
            bundle = response.json()
            entries = bundle.get('entry', [])
            
            return entries[0]['resource'] if entries else None
        except requests.RequestException:
            # MedicationKnowledge might not be available, return None
            return None
    
    def _parse_drug_info(self, medication_data: Dict, knowledge_data: Optional[Dict] = None) -> DrugInfo:
        """
        Parse FHIR resources to extract drug information.
        
        Args:
            medication_data (Dict): Medication resource data
            knowledge_data (Optional[Dict]): MedicationKnowledge resource data
            
        Returns:
            DrugInfo: Parsed drug information
        """
        # Extract drug name
        name = self._extract_drug_name(medication_data)
        
        # Extract active ingredients
        active_ingredients = self._extract_active_ingredients(medication_data)
        
        # Extract excipients (inactive ingredients)
        excipients = self._extract_excipients(medication_data)
        
        # Extract contraindications and cross-reactivity from knowledge data
        contraindications = []
        cross_reactivity = []
        
        if knowledge_data:
            contraindications = self._extract_contraindications(knowledge_data)
            cross_reactivity = self._extract_cross_reactivity(knowledge_data)
        
        return DrugInfo(
            name=name,
            active_ingredients=active_ingredients,
            excipients=excipients,
            contraindications=contraindications,
            cross_reactivity=cross_reactivity
        )
    
    def _extract_drug_name(self, medication_data: Dict) -> str:
        """Extract drug name from Medication resource."""
        code = medication_data.get('code', {})
        coding = code.get('coding', [])
        
        if coding:
            return coding[0].get('display', 'Unknown Drug')
        
        return medication_data.get('text', {}).get('div', 'Unknown Drug')
    
    def _extract_active_ingredients(self, medication_data: Dict) -> List[str]:
        """Extract active ingredients from Medication resource."""
        ingredients = []
        ingredient_list = medication_data.get('ingredient', [])
        
        for ingredient in ingredient_list:
            if ingredient.get('isActive', True):  # Default to active if not specified
                item_reference = ingredient.get('itemCodeableConcept', {})
                coding = item_reference.get('coding', [])
                if coding:
                    ingredients.append(coding[0].get('display', 'Unknown Ingredient'))
        
        return ingredients
    
    def _extract_excipients(self, medication_data: Dict) -> List[str]:
        """Extract excipients (inactive ingredients) from Medication resource."""
        excipients = []
        ingredient_list = medication_data.get('ingredient', [])
        
        for ingredient in ingredient_list:
            if not ingredient.get('isActive', True):  # Inactive ingredients
                item_reference = ingredient.get('itemCodeableConcept', {})
                coding = item_reference.get('coding', [])
                if coding:
                    excipients.append(coding[0].get('display', 'Unknown Excipient'))
        
        return excipients
    
    def _extract_contraindications(self, knowledge_data: Dict) -> List[str]:
        """Extract contraindications from MedicationKnowledge resource."""
        contraindications = []
        contraindication_list = knowledge_data.get('contraindication', [])
        
        for contraindication in contraindication_list:
            coding = contraindication.get('coding', [])
            if coding:
                contraindications.append(coding[0].get('display', 'Unknown Contraindication'))
        
        return contraindications
    
    def _extract_cross_reactivity(self, knowledge_data: Dict) -> List[str]:
        """Extract cross-reactivity information from MedicationKnowledge resource."""
        cross_reactivity = []
        
        # Cross-reactivity might be stored in different places depending on the FHIR implementation
        # Check in kinetics or other relevant sections
        kinetics = knowledge_data.get('kinetics', [])
        for kinetic in kinetics:
            # This is a simplified example - actual implementation would depend on 
            # how cross-reactivity data is structured in your FHIR server
            if 'crossReactivity' in kinetic:
                cross_reactivity.extend(kinetic['crossReactivity'])
        
        return cross_reactivity


# Example usage:
if __name__ == "__main__":
    # Configuration for FHIR server
    fhir_config = {
        'base_url': 'https://hapi.fhir.org/baseR4',
        'timeout': 30,
        'headers': {
            'Accept': 'application/fhir+json',
            'Content-Type': 'application/fhir+json'
        }
    }
    
    # Create FHIR database management instance
    fhir_db = FHIRDatabaseManagement(fhir_config)
    
    # Search for a drug
    try:
        drug_info = fhir_db.search_drug('aspirin')
        if drug_info:
            print(f"Drug Name: {drug_info.name}")
            print(f"Active Ingredients: {drug_info.active_ingredients}")
            print(f"Excipients: {drug_info.excipients}")
            print(f"Contraindications: {drug_info.contraindications}")
            print(f"Cross Reactivity: {drug_info.cross_reactivity}")
        else:
            print("Drug not found")
    except Exception as e:
        print(f"Error: {e}")