import cog
from pathlib import Path
from inference import Transformer
from utils import read_image
import cv2
import tempfile
from utils.image_processing import resize_image, normalize_input, denormalize_input
import numpy as np


class Predictor(cog.Predictor):
    def setup(self):
        pass

    @cog.input("model", type=str, default='Hayao', options=['Hayao', 'Shinkai'], help="Anime style model")
    @cog.input("image", type=Path, help="input image")
    def predict(self, image, model='Hayao'):
        transformer = Transformer(model)
        img = read_image(str(image))
        anime_img = transformer.transform(resize_image(img))[0]
        anime_img = denormalize_input(anime_img, dtype=np.int16)
        out_path = Path(tempfile.mkdtemp()) / "out.png"
        cv2.imwrite(str(out_path), anime_img[..., ::-1])
        return out_path
