import numpy as np
import os
import io
import pathlib
import time
import sys

from functools import partial
from PIL import Image
import pdf2image
from fastapi import *
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from typing import List
from pydantic import BaseModel

from dotenv import load_dotenv
from tritonclient import grpc as grpcclient
from tritonclient import http as httpclient
from tritonclient.utils import InferenceServerException
from PIL import ImageFont, ImageDraw, Image

from python.craftdet.utils import client

#############
# Initialize
#############
load_dotenv()
#
MODEL_NAME    = os.getenv("MODEL_NAME")
MODEL_VERSION = os.getenv("MODEL_VERSION", "")
BATCH_SIZE    = int(os.getenv("BATCH_SIZE", 1))
#
TRITON_URL           = os.getenv("TRITON_URL", "localhost:8000")
PROTOCAL      = os.getenv("PROTOCOL", "HTTP")
VERBOSE       = os.getenv("VERBOSE", "False").lower() in ("true", "1", "t")
ASYNC_SET     = os.getenv("ASYNC_SET", "False").lower() in ("true", "1", "t")
#

##################################
"""Define images app to store image after process"""
OUTPUT_DIR = os.getenv('OUTPUT_DIR', default='prediction')
os.makedirs(OUTPUT_DIR, exist_ok=True)

ORIGIN_IMAGE_PATH = os.getenv('ORIGIN_IMAGE_PATH', default='origin_images')
ORIGIN_IMAGE_PATH = pathlib.Path(OUTPUT_DIR + "/" + ORIGIN_IMAGE_PATH)
ORIGIN_IMAGE_PATH.mkdir(exist_ok=True)

OCR_IMAGE_PATH = os.getenv('OCR_IMAGE_PATH', default='ocr_images')
OCR_IMAGE_PATH = pathlib.Path(OUTPUT_DIR + "/" + OCR_IMAGE_PATH)
OCR_IMAGE_PATH.mkdir(exist_ok=True)

OCR_TEXT_PATH = os.getenv('OCR_TEXT_PATH', default='ocr_text')
OCR_TEXT_PATH = pathlib.Path(OUTPUT_DIR + "/" + OCR_TEXT_PATH)
OCR_TEXT_PATH.mkdir(exist_ok=True)


############
# Config
############

try:
    if PROTOCAL.lower() == "grpc":
        # Create gRPC client for communicating with the server
        triton_client = grpcclient.InferenceServerClient(
            url=TRITON_URL, verbose=VERBOSE
        )
    else:
        # Specify large enough concurrency to handle the number of requests.
        concurrency = 20 if ASYNC_SET else 1
        triton_client = httpclient.InferenceServerClient(
            url=TRITON_URL, verbose=VERBOSE, concurrency=concurrency
        )
except Exception as e:
    print("client creation failed: " + str(e))
    sys.exit(1)

try:
    model_metadata = triton_client.get_model_metadata(
        model_name=MODEL_NAME, model_version=MODEL_VERSION
    )
    model_config = triton_client.get_model_config(
        model_name=MODEL_NAME, model_version=MODEL_VERSION
    )
except InferenceServerException as e:
    print("failed to retrieve model metadata: " + str(e))
    sys.exit(1)

if PROTOCAL.lower() == "grpc":
    model_config = model_config.config
else:
    model_metadata, model_config = client.convert_http_metadata_config(
        model_metadata, model_config
    )

# parsing information of model
max_batch_size, input_name, output_name, format, dtype = client.parse_model(
    model_metadata, model_config
)

supports_batching = max_batch_size > 0
if not supports_batching and BATCH_SIZE != 1:
    print("ERROR: This model doesn't support batching.")
    sys.exit(1)

class ImageBatchRequest(BaseModel):
    images: List[np.ndarray]


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

    
    # for i, img in enumerate(images):
        
    #     img_rectify = np.ascontiguousarray(img)
        
    #     # predict OCR
    #     img_ocr, texts = ocr_predict(img_rectify, detector, ocr_model, i, len(images)) 

    #     # save to image api
    #     cv2.imwrite(f"{ORIGIN_IMAGE_PATH}/{file.filename}_{i}.jpg", np.asarray(img))
    #     cv2.imwrite(f"{OCR_IMAGE_PATH}/{file.filename}_{i}.jpg", img_ocr)
    #     with open(f"{OCR_TEXT_PATH}/{file.filename}_content{i}.txt", 'w') as f:
    #         for line in texts:
    #             f.write("%s\n" % line)


@app.post("/detect")
async def detect(request: ImageBatchRequest):
    #TODO: pipeline from upload file to image ocr return
    # 1. get file from upload
    # 2. call request to upload api
    # create folder for each file to save image
    
    images = request.images    
    assert len(images) > 0, "No image found after processing"

    # call request rieng cho tung module
    
@app.post("/detect/vietocr")
def vietocr(request: ImageBatchRequest):
    # image crops
    inputs, outputs = requestGenerator(request.images, "IMAGE",  ["OUTPUT"], np.float32)
    # Perform inference
    if PROTOCAL.lower() == "grpc":
        user_data = client.UserData()
        response = triton_client.async_infer(
            MODEL_NAME,
            inputs,
            partial(client.completion_callback, user_data),
            model_version=MODEL_VERSION,
            outputs=outputs,
        )
    else:
        async_request = triton_client.async_infer(
            MODEL_NAME,
            inputs,
            model_version=MODEL_VERSION,
            outputs=outputs,
        )
    
    # Collect results from the ongoing async requests
    if PROTOCAL.lower() == "grpc":
        (response, error) = user_data._completed_requests.get()
        if error is not None:
            return {"Error": "Inference failed with error: " + str(error)}
    else:
        # HTTP
        response = async_request.get_result()
    
    # Process the results
    # get output and return response
    outputs = response.as_numpy("SENTENCE")
    return JSONResponse(content={"sentence": outputs}, status_code=status.HTTP_200_OK)



    
@app.post("/detect/craftdet")
def craftdet(request: ImageBatchRequest):
    #TODO:  
    inputs, outputs = requestGenerator(request.images, "IMAGE", ["285", "onnx:Conv_275"], np.float32)
    # Perform inference
    try:
        start_time = time.time()

        if PROTOCAL.lower() == "grpc":
            user_data = client.UserData()
            response = triton_client.async_infer(
                MODEL_NAME,
                inputs,
                partial(client.completion_callback, user_data),
                model_version=MODEL_VERSION,
                outputs=outputs,
            )
        else:
            async_request = triton_client.async_infer(
                MODEL_NAME,
                inputs,
                model_version=MODEL_VERSION,
                outputs=outputs,
            )
    except InferenceServerException as e:
        return {"Error": "Inference failed with error: " + str(e)}

    # Collect results from the ongoing async requests
    if PROTOCAL.lower() == "grpc":
        (response, error) = user_data._completed_requests.get()
        if error is not None:
            return {"Error": "Inference failed with error: " + str(error)}
    else:
        # HTTP
        response = async_request.get_result()

    # Process the results    
    end_time = time.time()
    print("Process time: ", end_time - start_time)
    
    # get output and return response
    # Get outputs
    outputs1 = response.as_numpy("boxes")
    outputs2 = response.as_numpy("boxes_as_ratio")
    return JSONResponse(content={"boxes": outputs1, "boxes_as_ratio": outputs2})



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

def requestGenerator(batched_image_data, input_name, output_names, dtype):
    
    if PROTOCAL == "grpc":
        client = grpcclient
    else:
        client = httpclient

    # Set the input data
    inputs = [client.InferInput(input_name, batched_image_data.shape, dtype)]
    inputs[0].set_data_from_numpy(batched_image_data)

    outputs = [
        client.InferRequestedOutput(output_names[0], binary_data=True), 
        client.InferRequestedOutput(output_names[1], binary_data=True)
    ]

    return inputs, outputs

def cv2drawboxtext(img: np.ndarray, text, a):
        
    font = ImageFont.truetype("font-times-new-roman/SVN-Times New Roman 2.ttf", 20)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    # https://www.blog.pythonlibrary.org/2021/02/02/drawing-text-on-images-with-pillow-and-python/
    bbox = draw.textbbox(a, text, font=font, anchor='ls')

    draw.rectangle(a, fill="yellow", width=2) # draw bbox detection 
    draw.rectangle(bbox, fill="yellow") # draw text detection
    draw.text(a, text, font=font, anchor='ls', fill="black")
    return img