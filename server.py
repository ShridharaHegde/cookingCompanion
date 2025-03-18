from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import uvicorn
import cv2
import numpy as np
from script import driver

app = FastAPI()

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return StreamingResponse(driver(img), media_type="text/plain", headers={"X-Accel-Buffering": "no"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
