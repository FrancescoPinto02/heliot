from pydantic import BaseModel


class AllergyCheckEnhancedRequest(BaseModel):
    patient_id: str 
    drug_code: str 
    clinical_notes: str
    store: bool = False