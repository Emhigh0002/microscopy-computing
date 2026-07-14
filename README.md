# Microscopy AI: Microorganism Detection & Diagnostic Platform

A production-ready, research-grade clinical microscopy platform designed for automated microorganism detection, counting, shape-based measurement, and reporting. Built with FastAPI (Python), PyTorch, OpenCV, React, and Tailwind CSS.

## 🌟 Core Features

- **Microscope Image Analysis**: Upload slides and automatically segment and detect cells (e.g. *E. coli*, *Staph. aureus*, *C. albicans*, etc.).
- **Physical Size Calibration**: Compute area (µm²) and perimeter (µm) based on configurable lens magnification metrics.
- **Explainable AI (XAI)**: Generates simulated Grad-CAM/SHAP activation heatmaps outlining the diagnostic features the model identified.
- **Annotation Tool**: Full editor to draw bounding boxes, draw polygon masks, select/delete shapes, or adjust label classes. Corrected labels are saved to retrain the AI.
- **Interactive Chat Assistant**: Microbiology LLM assistant providing staining protocols, diagnostic distinction guidelines, and antibiotic recommendations with formal clinical citations.
- **Export Reports**: Generate clinical reports in PDF, XLSX (Excel), or CSV formats listing counts, average measurements, and confidence profiles.
- **Model Retraining**: Built-in retraining pipeline allowing pathologists to trigger model retraining on user-corrected dataset annotations.

---

## 🛠️ Tech Stack & Directory Structure

```text
microscopy_platform/
├── docker-compose.yml
├── README.md
└── backend/
    ├── requirements.txt
    ├── Dockerfile
    ├── app/
    │   ├── main.py                # FastAPI entry
    │   ├── database.py            # SQLite / PostgreSQL Session local
    │   ├── models.py              # SQLAlchemy DB Schema
    │   ├── schemas.py             # Pydantic validation schemas
    │   ├── core/                  # Configurations and JWT security
    │   ├── api/                   # Auth, Images, Predictions, Reports routers
    │   └── services/              # Inference, XAI, Assistant, Exporters
    └── static/
        └── index.html             # React + Tailwind + Plotly SPA Client
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Pip (Python Package Installer)
- *Optional: Docker & Docker Compose (for PostgreSQL/production deployment)*

### Method 1: Local Installation (Fastest)

1. **Navigate to the Backend directory:**
   ```bash
   cd backend
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the FastAPI Server:**
   ```bash
   uvicorn app.main:app --reload
   ```

5. **Access the Dashboard:**
   Open your browser and navigate to: **[http://localhost:8000](http://localhost:8000)**
   
   *Note: On startup, the database is automatically created and seeded with a default user:*
   - **Login Email:** `researcher@laboratory.org`
   - **Password:** `password123`

6. **Interactive API Documentation:**
   Available at **[http://localhost:8000/docs](http://localhost:8000/docs)**.

---

### Method 2: Docker Compose (Production Setup)

For full integration with a PostgreSQL database and containerized environment:

1. **Build and Run Services:**
   ```bash
   docker-compose up --build
   ```

2. **Database Storage**:
   The PostgreSQL instance compiles inside the `db` service. Persistent storage is automatically managed via docker volumes (`pg-data`).

3. **Application Port**:
   Access the dashboard at `http://localhost:8000`.

---

## 📖 Platform User Guide

### 1. Uploading Slides
- Navigate to the **Upload Image** tab.
- Choose your microscope slide image.
- Set the **Micron Calibration Scale (µm/pixel)**:
  - Higher lens magnifications (e.g. 100x oil immersion) require smaller scales (e.g. `0.02` µm/pixel).
  - Lower magnifications (e.g. 10x) use larger values (e.g. `0.2` µm/pixel).
- Click **Process Slide**. The AI will segment the cells, classify species, and redirect you to the main queue.

### 2. Microscope Viewer & Annotator
- Select an image in the **Annotation Tool** dropdown.
- Use `Scroll Wheel` to Zoom in and out. Left click + Drag to Pan.
- Toggle visibility of Bounding Boxes, Segmentation outlines, or Label Tags using the controls panel.
- Click **Explain Predictions (XAI)** to trigger the Grad-CAM activation heatmap overlay.
- To correct a label, click the shape on the canvas, adjust its species tag or dimensions in the **Observations Inspector**, or click **Delete Observation**.
- To draw new shapes, switch the tool from "Select" to "Box" or "Polygon". Click the Canvas to draw points. For polygons, click **Close Path** once finished.
- Corrections are automatically synchronized with the backend database.

### 3. Training Pipeline
- Navigate to **Model Training**.
- Set your preferred epochs and click **Start Retraining Pipeline**.
- The backend will trigger a background thread to train on the corrected annotations. Inspect validation accuracy, training losses, and mAP curves live!
- A successful retraining builds a new version (e.g., `1.1.0`), archiving the older weights.

### 4. Chat Assistant
- Navigate to the **AI Assistant** tab.
- Type any question regarding microscopic staining, diseases, distinguishing traits, or antibiotic treatments (e.g., *"What staining method is recommended for Candida albicans?"*).
- The assistant retrieves matching data and cites reliable scientific sources.
