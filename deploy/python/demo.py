import os
from predict_system import SystemPredictor
from paddleclas.deploy.utils import config
import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_bbox_results(image, results, font_path="./utils/simfang.ttf"):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    font_size = 18
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

    color = (0, 102, 255)

    for result in results:
        # empty results
        if result["rec_docs"] is None:
            continue

        xmin, ymin, xmax, ymax = result["bbox"]
        text = "{}, {:.2f}".format(result["rec_docs"], result["rec_scores"])
        th = font_size
        tw = font.getlength(text)
        # tw = int(len(result["rec_docs"]) * font_size) + 60
        start_y = max(0, ymin - th)

        draw.rectangle([(xmin + 1, start_y), (xmin + tw + 1, start_y + th)],
                       fill=color)

        draw.text((xmin + 1, start_y), text, fill=(255, 255, 255), font=font)

        draw.rectangle([(xmin, ymin), (xmax, ymax)],
                       outline=(255, 0, 0),
                       width=2)
    return np.array(image)


args = config.parse_args()
config = config.get_config('configs/inference_general_1.1.yaml', [], show=True)
config["Global"]["use_openvino"] = False

cur_openvino = False
system_predictor = SystemPredictor(config)

def search(image, use_openvino):
    global cur_openvino
    global system_predictor
    global config
    if cur_openvino != use_openvino:
        cur_openvino = use_openvino
        config["Global"]["use_openvino"] = use_openvino
        print(config["Global"]["use_openvino"])
        system_predictor = SystemPredictor(config)

    # img = image[:, :, ::-1]
    output = system_predictor.predict(image)
    output = draw_bbox_results(image, output)
    return output


demo = gr.Interface(
    fn=search,
    inputs=['image', "checkbox"],
    outputs=['image'],
)
if __name__ == "__main__":
    demo.launch(server_name='10.3.233.99')