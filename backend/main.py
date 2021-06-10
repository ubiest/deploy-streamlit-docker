# backend/main.py

########
######## CODE from the style transfer deploy

import uuid

#import cv2
import uvicorn
from fastapi import File
from fastapi import FastAPI
from fastapi import UploadFile
#import numpy as np
from PIL import Image

################

######### ADDED Jaume (8 juny)
#import streamlit as st
#import matplotlib.pyplot as plt
#import numpy as np
#import numpy.ma as ma
import fastai
from fastai.vision.all import *

#import altair as alt
import config
import inference

#####################

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}

####### NEW CODE
@app.post("/classify")
async def get_image(file: UploadFile = File(...)):
    image = await Image.open(file.file)
    output, output_dict = inference.predict(image)
    fname = f"/storage/{str(uuid.uuid4())}.jpg"
    image.save(fname) ## Log the image
    return {"phrase": output, "values": output_dict}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8088)
