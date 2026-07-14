import cv2
import numpy as np
import time
import math
import random
import os
import uuid
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Image as DBImage, User, Prediction, Annotation, AuditLog
from app.api import deps
from app.core.config import settings

router = APIRouter()

# Simulated live tracking cell state
simulated_cells = []

def init_simulated_cells():
    global simulated_cells
    simulated_cells = []
    classes = [
        {"name": "Escherichia coli", "color": (241, 102, 99)},      # BGR for #6366f1 (Indigo)
        {"name": "Staphylococcus aureus", "color": (129, 185, 16)}, # BGR for #10b981 (Green)
        {"name": "Bacillus subtilis", "color": (11, 158, 245)},     # BGR for #f59e0b (Amber)
        {"name": "Candida albicans", "color": (153, 72, 236)},      # BGR for #ec4899 (Pink)
        {"name": "Spermatozoon", "color": (247, 85, 168)}          # BGR for #a855f7 (Purple)
    ]
    for i in range(10):
        cls = random.choice(classes)
        simulated_cells.append({
            "id": i,
            "name": cls["name"],
            "color": cls["color"],
            "x": random.randint(200, 400),
            "y": random.randint(120, 280),
            "vx": random.uniform(-3, 3),
            "vy": random.uniform(-3, 3),
            "size": random.randint(10, 20)
        })

def generate_mock_live_frame() -> np.ndarray:
    global simulated_cells
    if not simulated_cells:
        init_simulated_cells()
        
    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    frame[:] = (20, 24, 33) # dark bg
    
    # Draw field aperture
    cv2.circle(frame, (300, 200), 190, (30, 35, 45), -1)
    cv2.circle(frame, (300, 200), 190, (60, 65, 75), 2)
    
    for cell in simulated_cells:
        cell["x"] += cell["vx"]
        cell["y"] += cell["vy"]
        
        dx = cell["x"] - 300
        dy = cell["y"] - 200
        dist = math.hypot(dx, dy)
        if dist > 175:
            # Bounce back
            cell["vx"] = -cell["vx"] * 0.9 + random.uniform(-0.5, 0.5)
            cell["vy"] = -cell["vy"] * 0.9 + random.uniform(-0.5, 0.5)
            cell["x"] += cell["vx"] * 2
            cell["y"] += cell["vy"] * 2
            
        x, y = int(cell["x"]), int(cell["y"])
        size = cell["size"]
        color = cell["color"]
        
        if cell["name"] == "Escherichia coli" or cell["name"] == "Bacillus subtilis":
            cv2.ellipse(frame, (x, y), (size, size // 2), 30, 0, 360, color, -1)
        elif cell["name"] == "Spermatozoon":
            cv2.circle(frame, (x, y), 5, color, -1)
            # Tail
            tx = x - int(15 * math.cos(time.time() * 6))
            ty = y + int(10 * math.sin(time.time() * 12))
            cv2.line(frame, (x, y), (tx, ty), color, 1)
        else:
            cv2.circle(frame, (x, y), size // 2, color, -1)
            
        # Draw bounding boxes and labels
        bx, by = x - size - 2, y - size - 2
        bw, bh = size * 2 + 4, size * 2 + 4
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 1)
        cv2.putText(frame, f"{cell['name']} 95%", (bx, by - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        
    # Crosshair
    cv2.line(frame, (300, 185), (300, 215), (75, 80, 90), 1)
    cv2.line(frame, (285, 200), (315, 200), (75, 80, 90), 1)
    
    return frame

def frame_generator(camera_id: int):
    cap = cv2.VideoCapture(camera_id)
    try:
        if not cap.isOpened():
            # Stream simulated live microscope view
            while True:
                frame = generate_mock_live_frame()
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.04) # ~25 FPS
        else:
            while True:
                ret, frame = cap.read()
                if not ret:
                    # Loop back or delay
                    time.sleep(0.1)
                    continue
                # Add a simulated AI label layer on top of raw USB camera stream if desired
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        cap.release()

@router.get("/list")
def list_connected_cameras(
    current_user: User = Depends(deps.get_current_active_user)
):
    """
    Scans and lists active USB microscope/camera devices.
    """
    cameras = []
    # Test indices 0, 1, 2
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append({
                "id": i,
                "name": f"Digital Microscope Camera (USB-{i})",
                "status": "Connected"
            })
            cap.release()
            
    if not cameras:
        # Provide simulated camera if none connected
        cameras.append({
            "id": 0,
            "name": "Simulated AI Microscope Feed (Virtual-0)",
            "status": "Connected"
        })
    return cameras

@router.get("/stream/{camera_id}")
def stream_camera_feed(
    camera_id: int,
    current_user: User = Depends(deps.get_current_active_user)
):
    """
    Streams live M-JPEG frames from the selected digital microscope.
    """
    return StreamingResponse(
        frame_generator(camera_id), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@router.post("/capture/{camera_id}")
def capture_camera_frame(
    camera_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(deps.get_current_active_user)
):
    """
    Captures a frame from the live stream and registers it in the platform directory.
    """
    cap = cv2.VideoCapture(camera_id)
    frame = None
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        
    if frame is None:
        # Generate simulated snapshot
        frame = generate_mock_live_frame()
        
    # Save snap
    unique_filename = f"capture_{uuid.uuid4()}.jpg"
    file_path = os.path.join(settings.UPLOAD_DIR, unique_filename)
    cv2.imwrite(file_path, frame)
    h, w, _ = frame.shape
    
    db_image = DBImage(
        user_id=current_user.id,
        file_name=f"Capture_Camera_{camera_id}_{int(time.time())}.jpg",
        file_path=file_path,
        scale_microns_px=0.05, # Default 40x calibration scale
        width=w,
        height=h,
        media_type="image",
        status="Uploaded"
    )
    db.add(db_image)
    db.commit()
    db.refresh(db_image)
    
    # Audit log
    db.add(AuditLog(
        user_id=current_user.id,
        action="Camera Capture",
        details=f"Captured frame from Camera {camera_id} saved as {db_image.id}"
    ))
    db.commit()
    
    return {"message": "Frame captured successfully", "image_id": db_image.id}
