from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import io
import picamera
import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image


class Model:

    def __init__(self):
        from MobileNetV2 import mobilenet_v2
        self.model = mobilenet_v2(pretrained=True).eval()
        print (self.model)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.convert_func = transforms.Compose([transforms.ToTensor(), normalize])
        with open('labels.txt', 'r') as f:
            self.labels =  [_.strip() for _ in f]

    def classify_image(self, image):
        image = np.array(image, dtype=np.float32) / 255.
        image = self.convert_func(image)
        with torch.no_grad():
            outputs = self.model(image[None, :])
        output = outputs[0].numpy()
        index = np.argsort(-output)[:5]
        plabels = [self.labels[i] for i in index]
        print (output[index], plabels)
        return output[index], plabels

    def __call__(self, image):
        return self.classify_image(image)


def main():
    model = Model()
    with picamera.PiCamera(resolution=(640, 480), framerate=30) as camera:
        camera.start_preview()
        try:
            stream = io.BytesIO()
            for _ in camera.capture_continuous(
                    stream, format='jpeg', use_video_port=True):
                stream.seek(0)
                image = Image.open(stream).convert('RGB').resize((224, 224), Image.ANTIALIAS)
                start_time = time.time()
                scores, labels = model(image)
                elapsed_ms = (time.time() - start_time) * 1000
                stream.seek(0)
                stream.truncate()
                result_str = ''
                for score, label in zip(scores, labels):
                    result_str += '%s %.2f\n' % (label, score) 
                camera.annotate_text = '%s\n%.1fms' % (result_str, elapsed_ms)
        finally:
            camera.stop_preview()


if __name__ == '__main__':
    main()
