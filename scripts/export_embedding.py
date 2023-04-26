import torch
import numpy as np
import cv2

import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic

from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.onnx import SamOnnxModel

checkpoint = 'sam_vit_h_4b8939.pth'
model_type = 'vit_h'

image = cv2.imread('../demo/src/assets/data/dogs.jpg')

start = time()
sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device='cuda')
predictor = SamPredictor(sam)
predictor.set_image(image)
image_embedding = predictor.get_image_embedding().cpu().numpy()
np.save('dogs_embedding.npy', image_embedding)

# quantize_dynamic(
#     model_input='sam_onnx_example.onnx',
#     model_output='sam_onnx_quantized_example.onnx',
#     optimize_model=True,
#     per_channel=False,
#     reduce_range=False,
#     weight_type=QuantType.QUInt8,
# )
