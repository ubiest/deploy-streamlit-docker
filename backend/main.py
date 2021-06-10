# backend/main.py

import uuid
import uvicorn
from fastapi import File
from fastapi import FastAPI
from fastapi import UploadFile
from PIL import Image
import fastai
from fastai.vision.all import *
import inference

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}

####### NEW CODE
@app.post("/classify")
async def get_image(file: UploadFile = File(...)):
    #image = await Image.open(file.file)
    image = Image.open(file.file)
    output, output_dict = inference.predict(image)
    fname = f"/storage/{str(uuid.uuid4())}.jpg"
    image.save(fname) ## Log the image
    return {"phrase": output, "values": output_dict}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
