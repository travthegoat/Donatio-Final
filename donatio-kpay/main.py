from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from kpay_processor import KPaySlipProcessor
import shutil
import os
from uuid import uuid4
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(name="KPay_Slip")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_ext = file.filename.split('.')[-1]
        filename = f"{uuid4()}.{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        processor = KPaySlipProcessor(file_path)
        result = processor.process()

        os.remove(file_path)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
    
    