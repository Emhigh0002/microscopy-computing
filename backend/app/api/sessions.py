import os
import cv2
import time
import math
import base64
import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from app.database import get_db, SessionLocal
from app.models import DetectionSession, SessionFrame, User, AuditLog, ClipExport
from app.schemas import DetectionSessionCreate, DetectionSessionResponse, SessionFrameCreate, SessionFrameResponse, ClipExportResponse
from app.api import deps
from app.core.config import settings
from app.services.inference import inference_service, ULTRALYTICS_AVAILABLE
from jose import jwt

logger = logging.getLogger("uvicorn.error")

router = APIRouter()

# Global Centroid Tracker for fallbacks
class CentroidTracker:
    def __init__(self):
        self.next_id = 1
        self.objects = {}  # id -> (cx, cy)
        
    def update(self, rects):
        new_objects = {}
        for rect in rects:
            x, y, w, h = rect
            cx, cy = x + w // 2, y + h // 2
            
            min_dist = 99999
            best_id = None
            for oid, opt in self.objects.items():
                dist = math.hypot(cx - opt[0], cy - opt[1])
                if dist < min_dist and dist < 50:
                    min_dist = dist
                    best_id = oid
            
            if best_id is not None:
                new_objects[best_id] = (cx, cy)
            else:
                new_objects[self.next_id] = (cx, cy)
                self.next_id += 1
                
        self.objects = new_objects
        return self.objects

# Configuration constants
MAX_CONCURRENT_SESSIONS = int(os.getenv("MAX_CONCURRENT_SESSIONS", 2))
INFERENCE_FPS = int(os.getenv("INFERENCE_FPS", 5))  # Configurable processing rate
SAVE_FRAME_INTERVAL = int(os.getenv("SAVE_FRAME_INTERVAL", 5))  # DB write throttling

@router.post("/start", response_model=DetectionSessionResponse, status_code=status.HTTP_201_CREATED)
def start_session(
    payload: DetectionSessionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(deps.get_current_active_user)
):
    """
    Starts a new real-time camera microscope detection session.
    Checks resource limits and blocks if MAX_CONCURRENT_SESSIONS is exceeded.
    """
    # 1. Enforce Concurrent Live Sessions Cap
    active_sessions_count = db.query(DetectionSession).filter(DetectionSession.status == "active").count()
    if active_sessions_count >= MAX_CONCURRENT_SESSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Resource Limit Reached: Cap of {MAX_CONCURRENT_SESSIONS} concurrent live sessions exceeded."
        )

    # 2. Get currently active model to bind to this session
    from app.models import Model
    active_model = db.query(Model).filter(Model.status == "active").first()
    model_id = payload.model_id or (active_model.id if active_model else None)

    # 3. Create Session Record
    session = DetectionSession(
        user_id=current_user.id,
        model_id=model_id,
        camera_source=payload.camera_source or "0",
        status="active",
        started_at=datetime.utcnow(),
        class_counts={},
        avg_confidence=0.0,
        total_frames_processed=0,
        total_frames_dropped=0,
        total_detections=0,
        achieved_inference_fps=0.0
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    # 4. Write Audit Log
    db.add(AuditLog(
        user_id=current_user.id,
        action="start_detection_session",
        details={"session_id": session.id, "camera_source": session.camera_source, "model_id": session.model_id}
    ))
    db.commit()

    return session

@router.post("/{session_id}/stop", response_model=DetectionSessionResponse)
def stop_session(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(deps.get_current_active_user)
):
    """
    Ends the live detection session, stops video processing, and records session summary metrics.
    """
    session = db.query(DetectionSession).filter(DetectionSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Detection session not found")
    if session.status != "active":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Session is already stopped")

    session.status = "completed"
    session.ended_at = datetime.utcnow()

    # Calculate achieved FPS
    duration = (session.ended_at - session.started_at).total_seconds()
    if duration > 0 and session.total_frames_processed > 0:
        session.achieved_inference_fps = round(session.total_frames_processed / duration, 2)

    db.add(AuditLog(
        user_id=current_user.id,
        action="stop_detection_session",
        details={"session_id": session.id, "duration_seconds": duration, "total_detections": session.total_detections}
    ))
    db.commit()
    db.refresh(session)
    return session

@router.get("/", response_model=List[DetectionSessionResponse])
def list_sessions(
    db: Session = Depends(get_db),
    current_user: User = Depends(deps.get_current_active_user)
):
    """
    Lists past microscope live detection sessions.
    """
    return db.query(DetectionSession).order_by(DetectionSession.started_at.desc()).all()

@router.get("/{session_id}", response_model=DetectionSessionResponse)
def get_session(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(deps.get_current_active_user)
):
    """
    Gets full details of a specific microscope detection session.
    """
    session = db.query(DetectionSession).filter(DetectionSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Detection session not found")
    return session

@router.get("/{session_id}/frames", response_model=List[SessionFrameResponse])
def get_session_frames(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(deps.get_current_active_user)
):
    """
    Retrieves all logged frames and detections captured during a session.
    """
    return db.query(SessionFrame).filter(SessionFrame.session_id == session_id).order_by(SessionFrame.timestamp.asc()).all()

@router.websocket("/{session_id}/stream")
async def websocket_stream(
    websocket: WebSocket,
    session_id: str
):
    """
    WebSocket endpoint for real-time video inference frame streaming and telemetry.
    Grabs frames from the configured camera source, runs yolo/contour inference under lock,
    calculates tracking IDs, updates rolling DB stats, and streams JSON overlay + base64 image data.
    """
    await websocket.accept()
    
    # 1. Authorize connection via query params token
    token = websocket.query_params.get("token")
    db = SessionLocal()
    user = None
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email: str = payload.get("sub")
        user = db.query(User).filter(User.email == email).first()
    except Exception:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        db.close()
        return

    if not user:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        db.close()
        return

    # 2. Retrieve the active session
    session = db.query(DetectionSession).filter(DetectionSession.id == session_id).first()
    if not session or session.status != "active":
        await websocket.send_json({"error": "Invalid or inactive session"})
        await websocket.close()
        db.close()
        return

    # 3. Open Video Capture Source
    camera_source = session.camera_source
    try:
        source_device = int(camera_source)
    except ValueError:
        source_device = camera_source
        
    cap = cv2.VideoCapture(source_device)
    if not cap.isOpened():
        logger.error(f"Failed to open camera source: {camera_source}")
        await websocket.send_json({"error": f"Failed to detect or connect to digital microscope camera source: {camera_source}"})
        await websocket.close()
        db.close()
        return

    # 4. Start Ingestion and Inference Loop
    target_interval = 1.0 / INFERENCE_FPS
    last_inference_time = 0.0
    processed_timestamps = []
    
    # Centroid tracker fallback
    tracker = CentroidTracker()
    CLASSES_LIST = [c["name"] for c in settings.TARGET_CLASSES]

    try:
        while True:
            # Check if session was stopped by HTTP post
            db.refresh(session)
            if session.status != "active":
                logger.info("Session stopped. Ending WebSocket stream.")
                break

            # Grab frame to keep internal buffer clear
            ret = cap.grab()
            if not ret:
                logger.warning("Camera disconnected or end of feed reached.")
                await websocket.send_json({"error": "Microscope camera feed lost."})
                break

            current_time = time.time()
            if current_time - last_inference_time >= target_interval:
                # Retrieve and decode frame
                ret, frame = cap.retrieve()
                if not ret or frame is None:
                    break

                last_inference_time = current_time
                processed_timestamps.append(current_time)
                if len(processed_timestamps) > 30:
                    processed_timestamps.pop(0)

                # Resize frame to standard processing size
                h, w = frame.shape[:2]
                if w > 800:
                    frame = cv2.resize(frame, (800, int(800 * (h/w))))
                    h, w = frame.shape[:2]

                detections = []
                avg_conf_sum = 0.0

                # 5. Execute inference under model lock
                with inference_service.lock:
                    model_to_use = inference_service.yolo_model
                    
                if model_to_use is not None:
                    try:
                        # Built-in YOLO tracking (ByteTrack)
                        results = model_to_use.track(frame, persist=True, verbose=False)
                        for result in results:
                            boxes = result.boxes
                            for box in boxes:
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                conf = float(box.conf[0])
                                cls_idx = int(box.cls[0])
                                track_id = int(box.id[0]) if box.id is not None else -1
                                class_name = CLASSES_LIST[cls_idx] if cls_idx < len(CLASSES_LIST) else "Escherichia coli"

                                detections.append({
                                    "class": class_name,
                                    "confidence": round(conf, 2),
                                    "box": {"x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1)},
                                    "track_id": track_id
                                })
                                avg_conf_sum += conf
                    except Exception as inf_err:
                        logger.error(f"Inference error during session: {inf_err}")
                else:
                    # Fallback to OpenCV contours and CentroidTracker
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    rects = []
                    for cnt in contours:
                        if cv2.contourArea(cnt) > 40:
                            bx, by, bw, bh = cv2.boundingRect(cnt)
                            rects.append((bx, by, bw, bh))
                            
                    tracks = tracker.update(rects)
                    for track_id, (cx, cy) in tracks.items():
                        # Find matching rect
                        matching_rect = next((r for r in rects if r[0] <= cx <= r[0]+r[2] and r[1] <= cy <= r[1]+r[3]), None)
                        if matching_rect:
                            bx, by, bw, bh = matching_rect
                            detections.append({
                                "class": "Escherichia coli", # Default fallback
                                "confidence": 1.0,
                                "box": {"x": bx, "y": by, "w": bw, "h": bh},
                                "track_id": track_id
                            })
                            avg_conf_sum += 1.0

                # 6. Calculate rolling stats
                total_dets_in_frame = len(detections)
                session.total_frames_processed += 1
                session.total_detections += total_dets_in_frame

                if total_dets_in_frame > 0:
                    avg_conf_frame = avg_conf_sum / total_dets_in_frame
                    prev_count = session.total_detections - total_dets_in_frame
                    if session.total_detections > 0:
                        session.avg_confidence = round(
                            ((session.avg_confidence * prev_count) + avg_conf_sum) / session.total_detections, 2
                        )
                    else:
                        session.avg_confidence = round(avg_conf_frame, 2)

                    # Update rolling counts
                    counts = dict(session.class_counts or {})
                    for d in detections:
                        cls = d["class"]
                        counts[cls] = counts.get(cls, 0) + 1
                    session.class_counts = counts

                # Save Frame Image occasionally to conserve space
                frame_path = None
                if total_dets_in_frame > 0 and (session.total_frames_processed % SAVE_FRAME_INTERVAL == 0):
                    session_dir = os.path.join(settings.UPLOAD_DIR, "sessions", session_id)
                    os.makedirs(session_dir, exist_ok=True)
                    frame_filename = f"frame_{session.total_frames_processed}.jpg"
                    frame_path = os.path.join(session_dir, frame_filename)
                    cv2.imwrite(frame_path, frame)
                    
                    # Persist SessionFrame to DB
                    db_frame = SessionFrame(
                        session_id=session_id,
                        timestamp=datetime.utcnow(),
                        frame_path=frame_path,
                        detections=detections
                    )
                    db.add(db_frame)
                    
                db.commit()

                # Calculate actual achieved FPS
                achieved_fps = 0.0
                if len(processed_timestamps) > 1:
                    achieved_fps = len(processed_timestamps) / (processed_timestamps[-1] - processed_timestamps[0])

                # 7. Broadcast base64 image + detection data to client
                _, buffer = cv2.imencode(".jpg", frame)
                base64_frame = base64.b64encode(buffer).decode("utf-8")
                
                await websocket.send_json({
                    "frame": f"data:image/jpeg;base64,{base64_frame}",
                    "detections": detections,
                    "stats": {
                        "achieved_fps": round(achieved_fps, 1),
                        "total_detections": session.total_detections,
                        "class_counts": session.class_counts,
                        "avg_confidence": session.avg_confidence
                    }
                })

            await asyncio.sleep(0.005) # Yield thread briefly
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
    except Exception as err:
        logger.error(f"WebSocket execution error: {err}")
        try:
            await websocket.send_json({"error": str(err)})
        except:
            pass
    finally:
        cap.release()
        db.close()

@router.get("/{session_id}/export")
def export_session_log(
    session_id: str,
    format: str = Query("json", regex="^(json|csv)$"),
    db: Session = Depends(get_db),
    current_user: User = Depends(deps.get_current_active_user)
):
    """
    Exports the complete detection log for a session as CSV or JSON.
    """
    session = db.query(DetectionSession).filter(DetectionSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
        
    frames = db.query(SessionFrame).filter(SessionFrame.session_id == session_id).order_by(SessionFrame.timestamp.asc()).all()
    
    if format == "json":
        data = []
        for f in frames:
            for d in f.detections:
                data.append({
                    "timestamp": f.timestamp.isoformat(),
                    "class": d.get("class"),
                    "confidence": d.get("confidence"),
                    "box": d.get("box"),
                    "track_id": d.get("track_id")
                })
        return {"session_id": session_id, "detections": data}
    else:
        # Generate CSV output
        import io
        from fastapi.responses import StreamingResponse
        output = io.StringIO()
        output.write("Timestamp,Class,Confidence,Box_X,Box_Y,Box_W,Box_H,Track_ID\n")
        
        for f in frames:
            for d in f.detections:
                box = d.get("box", {})
                bx = box.get("x", 0)
                by = box.get("y", 0)
                bw = box.get("w", 0)
                bh = box.get("h", 0)
                output.write(f"{f.timestamp.isoformat()},{d.get('class')},{d.get('confidence')},{bx},{by},{bw},{bh},{d.get('track_id')}\n")
                
        output.seek(0)
        return StreamingResponse(
            io.BytesIO(output.read().encode("utf-8")),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=session_{session_id}_export.csv"}
        )

@router.get("/{session_id}/clip", response_model=ClipExportResponse)
def export_session_clip(
    session_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(deps.get_current_active_user)
):
    """
    Retrieves or triggers the background task to render and export an annotated HD video clip of the session frames.
    """
    session = db.query(DetectionSession).filter(DetectionSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
        
    # Check if a clip has already been triggered
    existing_clip = db.query(ClipExport).filter(ClipExport.session_id == session_id).order_by(ClipExport.created_at.desc()).first()
    if existing_clip:
        return existing_clip

    db_clip = ClipExport(
        user_id=current_user.id,
        session_id=session_id,
        file_name=f"Session_Clip_{session_id[:8]}.mp4",
        format="mp4",
        duration_seconds=5.0,
        draw_overlays=True,
        status="queued",
        progress=0.0
    )
    db.add(db_clip)
    db.commit()
    db.refresh(db_clip)
    
    background_tasks.add_task(process_background_clip_export, db_clip.id)
    
    # Audit log
    db.add(AuditLog(
        user_id=current_user.id,
        action="request_session_clip_export",
        details={"session_id": session_id, "clip_id": db_clip.id}
    ))
    db.commit()
    
    return db_clip

@router.post("/frame/{frame_id}/to-image")
def convert_frame_to_image(
    frame_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(deps.get_current_active_user)
):
    """
    Converts/imports a saved SessionFrame and its detections into the static Image/Annotation
    tables so they can be managed in the standard annotation correction and retraining pipeline.
    """
    from app.models import Image as DBImage, Annotation as DBAnnotation
    
    session_frame = db.query(SessionFrame).filter(SessionFrame.id == frame_id).first()
    if not session_frame:
        raise HTTPException(status_code=404, detail="Session frame not found")
        
    if not session_frame.frame_path or not os.path.exists(session_frame.frame_path):
        raise HTTPException(status_code=400, detail="Frame image file not found on disk")
        
    # Read image dimensions
    img = cv2.imread(session_frame.frame_path)
    h, w = (480, 640)
    if img is not None:
        h, w = img.shape[:2]
        
    # Create static Image record
    db_image = DBImage(
        user_id=current_user.id,
        file_name=f"session_frame_{frame_id[:8]}.jpg",
        file_path=session_frame.frame_path,
        width=w,
        height=h,
        scale_microns_px=0.05,  # Default scale
        status="Annotated"
    )
    db.add(db_image)
    db.commit()
    db.refresh(db_image)
    
    # Import annotations
    for d in session_frame.detections:
        box = d.get("box", {})
        if box:
            ann = DBAnnotation(
                image_id=db_image.id,
                label_class=d.get("class", "Escherichia coli"),
                coordinates={
                    "box": {
                        "x": int(box.get("x", 0)),
                        "y": int(box.get("y", 0)),
                        "w": int(box.get("w", 0)),
                        "h": int(box.get("h", 0))
                    }
                }
            )
            db.add(ann)
            
    db.commit()
    
    # Audit log
    db.add(AuditLog(
        user_id=current_user.id,
        action="convert_frame_to_static_image",
        details={"frame_id": frame_id, "image_id": db_image.id}
    ))
    db.commit()
    
    return {"message": "Frame successfully imported to main annotation pipeline", "image_id": db_image.id}
