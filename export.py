import os
import cv2
import torch
from movenet.opts import opts
from movenet.detectors.detector_factory import detector_factory
from alfred import print_shape
import cv2
import numpy as np

torch.set_grad_enabled(False)
torch.manual_seed(10234)

torch.set_grad_enabled(False)

image_ext = ["jpg", "jpeg", "png", "webp"]
video_ext = ["mp4", "mov", "avi", "mkv"]
time_stats = ["tot", "load", "pre", "net", "dec", "post", "merge"]

def pre_process(image, meta=None):
    mean = [ 1, 1, 1]
    std = [ 1, 1, 1]
    height, width = image.shape[0:2]

    # padding all images to be square.
    if height > width:
        diff = height - width
        image = cv2.copyMakeBorder(
            image,
            0,
            0,
            int(diff // 2),
            int(diff // 2 + diff % 2),
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
    elif height < width:
        diff = width - height
        image = cv2.copyMakeBorder(
            image,
            int(diff // 2),
            int(diff // 2 + diff % 2),
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

    new_height = 256  # 192
    new_width = 256  # 192

    inp_height = new_height
    inp_width = new_width
    c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
    s = np.array([inp_width, inp_height], dtype=np.float32)

    inp_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )
    inp_image = cv2.cvtColor(inp_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    inp_image = ((inp_image / 127.5 - mean) / std).astype(np.float32)
    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    images = torch.from_numpy(images)
    return images


def demo(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)

    model = detector.model
    model.eval()
    a = torch.randn([1, 3, 256, 256])
    a = cv2.imread('data/input_image.jpeg')
    a = pre_process(a)
    
    traced_m = torch.jit.trace(model, [a])
    traced_m.save("movenet.pt")
    torch.onnx.export(model, a, "movenet.onnx", opset_version=13)

    o = traced_m(a)

    a.numpy().tofile("data0.bin")
    o.numpy().tofile("gt.bin")

    print(o)
    print_shape(a, o)


if __name__ == "__main__":
    opt = opts().init()
    demo(opt)
