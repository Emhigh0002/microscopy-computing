from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import User, AuditLog
from app.schemas import AssistantRequest, AssistantResponse
from app.api import deps
from app.services.assistant import assistant_service

router = APIRouter()

@router.post("/query", response_model=AssistantResponse)
def query_assistant(
    payload: AssistantRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(deps.get_current_active_user)
):
    """
    Connects to the clinical AI assistant.
    Provides expert responses regarding staining, morphological traits, susceptibility,
    and microorganism identification with scientific citations.
    """
    ans = assistant_service.answer_microbiology_question(payload.query)
    
    # Audit log (log user queries for LIMS/clinical tracking)
    log = AuditLog(
        user_id=current_user.id,
        action="assistant_query",
        details={"query": payload.query[:200]}
    )
    db.add(log)
    db.commit()
    
    return ans
