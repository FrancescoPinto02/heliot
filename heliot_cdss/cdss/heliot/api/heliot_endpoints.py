from .models.heliot_models import *
from .services.heliot_llm import *
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

import os
from openai import OpenAI

router = APIRouter()


heliot = HeliotLLM()


@router.post("/allergy_check_enhanced")
async def allergy_check(request: AllergyCheckEnhancedRequest):
    patient_id = request.patient_id
    drug_code = request.drug_code
    clinical_notes = request.clinical_notes
    store = request.store

    return StreamingResponse(heliot.dss_check_enhanced(patient_id, drug_code, clinical_notes, store), media_type='text/event-stream')