import asyncio
import re
import sys
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input_videos" / "uploads"
OUTPUT_DIR = BASE_DIR / "output_videos"
WEB_DIR = BASE_DIR / "web"

ALLOWED_EXTENSIONS = {".avi", ".mov", ".mp4", ".mkv", ".webm"}

app = FastAPI(title="Basketball Video Analytics")


def safe_stem(filename: str) -> str:
    stem = Path(filename).stem or "video"
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", stem).strip(".-")
    return stem[:80] or "video"


def output_name_for(input_path: Path) -> str:
    video_name = input_path.stem
    if video_name.startswith("input_"):
        return video_name.replace("input_", "output_", 1)
    return f"output_{video_name}"


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/process")
async def process_video(file: UploadFile = File(...)):
    extension = Path(file.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Upload a video file with one of these extensions: AVI, MOV, MP4, MKV, WEBM.",
        )

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    upload_name = f"input_{safe_stem(file.filename)}_{uuid4().hex[:10]}{extension}"
    input_path = INPUT_DIR / upload_name
    output_name = output_name_for(input_path)
    output_path = OUTPUT_DIR / f"{output_name}.avi"

    try:
        with input_path.open("wb") as destination:
            while chunk := await file.read(1024 * 1024):
                destination.write(chunk)
    finally:
        await file.close()

    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(BASE_DIR / "main.py"),
        str(input_path),
        cwd=str(BASE_DIR),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    stdout_chunks = []
    while True:
        line = await process.stdout.readline()
        if not line:
            break
        # Print to the terminal in real-time
        sys.stdout.write(line.decode(errors="replace"))
        sys.stdout.flush()
        stdout_chunks.append(line)

    await process.wait()
    stdout_bytes = b"".join(stdout_chunks)
    stdout = stdout_bytes
    stderr = b""

    if process.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "The video pipeline failed.",
                "stdout": stdout.decode(errors="replace")[-4000:],
                "stderr": stderr.decode(errors="replace")[-4000:],
            },
        )

    if not output_path.exists():
        raise HTTPException(
            status_code=500,
            detail="The pipeline finished, but no annotated output video was found.",
        )

    return {
        "filename": output_path.name,
        "download_url": f"/api/download/{output_path.name}",
        "logs": stdout.decode(errors="replace")[-4000:],
    }


@app.get("/api/download/{filename}")
def download_video(filename: str):
    path = OUTPUT_DIR / Path(filename).name
    if not path.exists() or path.suffix.lower() != ".avi":
        raise HTTPException(status_code=404, detail="Output video not found.")

    return FileResponse(
        path,
        media_type="video/x-msvideo",
        filename=path.name,
    )


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
