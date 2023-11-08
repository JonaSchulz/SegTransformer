import segment_anything as sam
import numpy as np
from PIL import Image

with Image.open("../data/train_image_original/0001.png") as image:
    image = np.array(image)

model = sam.sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
model.to("cuda")
# predictor = sam.SamPredictor(model)
# predictor.set_image(image)

mask_generator = sam.SamAutomaticMaskGenerator(model)
masks = mask_generator.generate(image)

print(masks)
