import cv2
import numpy as np
import os
import io
import torch
import torch.nn.functional as F
import pathlib

from PIL import Image
from tqdm import tqdm
import pdf2image
from fastapi import *
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from typing import List
from pydantic import BaseModel

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from ocr.craftdet.detection import Detector
from ocr.preprocessor.model import DewarpTextlineMaskGuide
from ocr.utils import bbox2ibox, cv2crop, cv2drawbox


DEVICE = os.getenv('DEVICE', default="cuda:0")
IMAGE_SIZE = os.getenv('IMAGE_SIZE', default=224)


def ocr_predict(img_rectify, detector, ocr_model, idx, number_images):
    """idx, number_images: for tqdm progress bar"""
    texts = []
    z = detector.detect(img_rectify)

    batch_img_rectify_crop = []
    for j in tqdm(range(len(z['boxes'])), desc='Process page ({}/{})'.format(idx + 1, number_images)):
        ib = bbox2ibox(z['boxes'][j])
        img_rectify_crop = cv2crop(img_rectify, ib[0], ib[1])
        batch_img_rectify_crop.append(Image.fromarray(img_rectify_crop))
        img_rectify = cv2drawbox(img_rectify, ib[0], ib[1])

    texts = ocr_model.predict_batch(batch_img_rectify_crop)
    return img_rectify, texts

##################################
'''Load checkpoint and weight'''

# OCR
lsd = cv2.createLineSegmentDetector()
detector = Detector(
    craft=os.getcwd() + '/weights/craft/mlt25k.pth',
    refiner=os.getcwd() + '/weights/craft/refinerCTW1500.pth',
    use_cuda=True if "cuda" in DEVICE else False
)
#
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = str(os.getcwd() + '/weights/ocr/vgg_transformer.pth')
config['device'] = DEVICE
ocr_model = Predictor(config)

##################################
"""Define images app to store image after process"""
os.makedirs("./prediction", exist_ok=True)

ORIGIN_IMAGE_PATH = os.getenv('ORIGIN_IMAGE_PATH', default='origin_images')
ORIGIN_IMAGE_PATH = pathlib.Path("prediction/" + ORIGIN_IMAGE_PATH)
ORIGIN_IMAGE_PATH.mkdir(exist_ok=True)

OCR_IMAGE_PATH = os.getenv('OCR_IMAGE_PATH', default='ocr_images')
OCR_IMAGE_PATH = pathlib.Path("prediction/" + OCR_IMAGE_PATH)
OCR_IMAGE_PATH.mkdir(exist_ok=True)

OCR_TEXT_PATH = os.getenv('OCR_TEXT_PATH', default='ocr_text')
OCR_TEXT_PATH = pathlib.Path("prediction/" + OCR_TEXT_PATH)
OCR_TEXT_PATH.mkdir(exist_ok=True)


class ListPathRequest(BaseModel):
    paths: List[str]


#
image_api = FastAPI()

# 
@image_api.get("/")
def read_app():
    return {"Hello": "Image Apps"}

image_api.mount('/ori_imgs', StaticFiles(directory=str(ORIGIN_IMAGE_PATH)), name='origin_images')
image_api.mount('/ocr_imgs', StaticFiles(directory=str(OCR_IMAGE_PATH)), name='ocr_images')
image_api.mount('/ocr_texts', StaticFiles(directory=str(OCR_TEXT_PATH)), name='oct_text')

##################################
'''Main App'''

#
app = FastAPI()
app.mount("/imageapi", image_api)
1
#
@app.get("/")
def root():
    return {"Hello": "Main Apps"}
#
@app.post("/upload/")
async def upload(files: List[UploadFile] = File(...)) -> JSONResponse:
    for file in files:
        request_object_content = await file.read()
        extension = file.filename.split(".")[-1]

        images = []
        if extension in ["jpg", "jpeg", "png"]:
            images = [Image.open(io.BytesIO(request_object_content))]
        elif extension in ["pdf"]:
            images = pdf2image.convert_from_bytes(request_object_content)
        else:
            raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Unsupported file type")
        
        assert len(images) > 0, "No image found after processing"
        for idx, image in enumerate(images):
            image.save(f"{ORIGIN_IMAGE_PATH}/{file.filename}_image{idx}.jpg")

    return JSONResponse(content={
        "message": [file.file.name for file in files]
    }, status_code=status.HTTP_200_OK)


@app.get("/ocr")
def get_ocr():
    ocr_imgs = [f'/imageapi/ocr_imgs/{k}' for k in os.listdir(OCR_IMAGE_PATH)]
    return {"ocr": ocr_imgs}

@app.get("/texts")
def get_texts():
    ocr_texts = [f'/imageapi/ocr_texts/{k}' for k in os.listdir(OCR_TEXT_PATH)]
    return {"texts": ocr_texts}

@app.get("/origin")
def get_origin():
    origin_imgs = [f'/imageapi/ori_imgs/{k}' for k in os.listdir(ORIGIN_IMAGE_PATH)]
    return {"retify": origin_imgs}

##################################