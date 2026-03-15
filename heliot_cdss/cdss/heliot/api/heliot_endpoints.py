from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from .models.heliot_models import *
from .rate_limit.deps import require_rate_limit
from .services.api_key_service import AuthContext
from .services.heliot_llm import *

router = APIRouter()

heliot = HeliotLLM()

@router.post("/allergy_check_enhanced")
async def allergy_check(
    request: AllergyCheckEnhancedRequest,
    auth_context: AuthContext = Depends(require_rate_limit),
):
    patient_id = request.patient_id
    drug_code = request.drug_code
    clinical_notes = request.clinical_notes
    store = request.store

    return StreamingResponse(heliot.dss_check_enhanced(patient_id, drug_code, clinical_notes, store), media_type='text/event-stream')