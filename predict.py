from pathlib import Path
from inference import Predictor as MyPredictor
from utils import read_image
import cv2
import tempfile
from utils.image_processing import resize_image, normalize_input, denormalize_input
import numpy as np
from cog import BasePredictor, Path, Input


class Predictor(BasePredictor):
    def setup(self):
        pass

    def predict(
        self,
        image: Path = Input(description="Image"),
        model: str = Input(
            description="Style",
            default='Hayao:v2',
            choices=[
                'Hayao',
                'Shinkai',
                'Hayao:v2'
            ]
        )
    ) -> Path:
        version = model.split(":")[-1]
        predictor = MyPredictor(model, version)
        img = read_image(str(image))
        anime_img = predictor.transform(resize_image(img))[0]
        out_path = Path(tempfile.mkdtemp()) / "out.png"
        cv2.imwrite(str(out_path), anime_img[..., ::-1])
        return out_path

