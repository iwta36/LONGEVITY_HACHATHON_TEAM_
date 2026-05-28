"""
main.py — FastAPI wrapper for Virtual Screening Pipeline
=========================================================
Exposes three endpoints:

  POST /screen              Upload a PDB file → starts the pipeline
  GET  /status              Check progress of the running job
  GET  /results/download    Download top5_hits.csv when complete

One job runs at a time. A new request is rejected while a job is running.
The drugs/ folder lives next to this file on the Railway server.
"""

import threading
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from pipeline import VirtualScreeningPipeline


# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────

app = FastAPI(title="Virtual Screening Pipeline — Script 1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict to your Lovable URL after testing
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR  = Path(__file__).parent
DRUGS_DIR = BASE_DIR / "drugs"
JOBS_DIR  = BASE_DIR / "jobs"
JOBS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# JOB STATE
# One job at a time — no queue needed
# ─────────────────────────────────────────────

job = {
    "status"       : "idle",    # idle | running step names | complete | failed
    "progress"     : 0,         # 0 – 100
    "docked_so_far": 0,
    "total_drugs"  : 0,
    "started_at"   : None,
    "finished_at"  : None,
    "csv_path"     : None,      # Path to top5_hits.csv when complete
    "error"        : None,
}

# Thread lock — prevents two simultaneous runs
_lock = threading.Lock()


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/")
def health_check():
    return {
        "status" : "running",
        "service": "Virtual Screening Pipeline — Script 1",
        "job"    : job["status"],
    }


@app.post("/screen")
async def start_screening(
    background_tasks: BackgroundTasks,
    target_pdb: UploadFile = File(...),
):
    """
    Upload a PDB file to start the screening pipeline.
    Returns immediately with a confirmation.
    Poll /status to track progress.
    """
    # Reject if a job is already running
    if job["status"] not in ("idle", "complete", "failed"):
        raise HTTPException(
            status_code=409,
            detail=f"A job is already running ({job['status']}). "
                   f"Wait for it to finish before submitting a new one."
        )

    # Validate file extension
    if not target_pdb.filename.endswith(".pdb"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be a .pdb file."
        )

    # Save uploaded PDB to jobs folder
    job_dir  = JOBS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    job_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = job_dir / "target.pdb"

    with open(pdb_path, "wb") as f:
        f.write(await target_pdb.read())

    # Reset job state
    job.update({
        "status"       : "queued",
        "progress"     : 0,
        "docked_so_far": 0,
        "total_drugs"  : 0,
        "started_at"   : datetime.now().isoformat(),
        "finished_at"  : None,
        "csv_path"     : None,
        "error"        : None,
    })

    # Run pipeline in background so HTTP response returns immediately
    background_tasks.add_task(_run_pipeline, pdb_path, job_dir)

    return {
        "message" : "Job started. Poll /status for progress.",
        "pdb_file": target_pdb.filename,
    }


@app.get("/status")
def get_status():
    """
    Returns current job state.
    Poll this every few seconds from Lovable to update the progress bar.
    """
    return {
        "status"       : job["status"],
        "progress"     : job["progress"],
        "docked_so_far": job["docked_so_far"],
        "total_drugs"  : job["total_drugs"],
        "started_at"   : job["started_at"],
        "finished_at"  : job["finished_at"],
        "error"        : job["error"],
        "ready"        : job["status"] == "complete",
    }


@app.get("/results/download")
def download_results():
    """
    Download top5_hits.csv — only available when job status is 'complete'.
    This CSV is the direct input for Script 2.
    """
    if job["status"] != "complete":
        raise HTTPException(
            status_code=425,
            detail=f"Results not ready yet. Current status: {job['status']}"
        )

    csv_path = Path(job["csv_path"])
    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail="CSV file not found on server. The job may need to be rerun."
        )

    return FileResponse(
        path        = csv_path,
        media_type  = "text/csv",
        filename    = "top5_hits.csv",
    )


# ─────────────────────────────────────────────
# BACKGROUND PIPELINE RUNNER
# ─────────────────────────────────────────────

def _run_pipeline(pdb_path: Path, job_dir: Path):
    """
    Runs in a background thread.
    Updates the global job dict at each step so /status always reflects
    the current state.
    """
    with _lock:
        try:
            def on_progress(status, progress, docked, total):
                job["status"]        = status
                job["progress"]      = progress
                job["docked_so_far"] = docked
                job["total_drugs"]   = total

            pipeline = VirtualScreeningPipeline(
                pdb_file   = pdb_path,
                drugs_dir  = DRUGS_DIR,
                output_dir = job_dir / "output",
                work_dir   = job_dir / "workdir",
                on_progress= on_progress,
            )

            csv_path = pipeline.run()

            job["status"]     = "complete"
            job["progress"]   = 100
            job["csv_path"]   = str(csv_path)
            job["finished_at"]= datetime.now().isoformat()

        except Exception as e:
            job["status"]     = "failed"
            job["error"]      = str(e)
            job["finished_at"]= datetime.now().isoformat()