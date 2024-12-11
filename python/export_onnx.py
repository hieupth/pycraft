import os
import torch

from craftdet.models import BaseNet, CraftNet, RefineNet
from craftdet.models.utils import load_model

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

IMAGE_SIZE = os.getenv('IMAGE_SIZE', default=224)

os.makedirs("models", exist_ok=True)

use_cuda = torch.cuda.is_available()

torch_model = BaseNet(freeze=False, pretrained=False)

torch_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
onnx_program = torch.onnx.export(torch_model, torch_input, "models/basenet.onnx")
# onnx_program.save()


# craft: str = './weights/craft/mlt25k.pth'
# craft_net = load_model(CraftNet(), craft, use_cuda)
# torch_input = torch.randn((1, 3, IMAGE_SIZE, IMAGE_SIZE), device='cuda' if use_cuda else 'cpu')
# onnx_program = torch.onnx.dynamo_export(craft_net, torch_input)
# onnx_program.save("models/craftnet.onnx")


# refiner: str = './weights/craft/refinerCTW1500.pth'
# refine_net = load_model(RefineNet(), refiner, use_cuda)
# torch_input = torch.randn((1, 3, IMAGE_SIZE, IMAGE_SIZE), device='cuda' if use_cuda else 'cpu')
# onnx_program = torch.onnx.dynamo_export(refine_net, torch_input)
# onnx_program.save("models/refinet.onnx")



# config = Cfg.load_config_from_name('vgg_transformer')
# config['weights'] = str(os.getcwd() + './weights/ocr/vgg_transformer.pth')
# config['device'] = 'cpu' if not use_cuda else 'cuda:0'
# ocr_model = Predictor(config)
# torch_input = torch.randn((1, 3, IMAGE_SIZE, IMAGE_SIZE), device='cuda' if use_cuda else 'cpu')
# onnx_program = torch.onnx.dynamo_export(ocr_model.model, torch_input)
# onnx_program.save("models/ocr.onnx")

# print("Models exported successfully.")

