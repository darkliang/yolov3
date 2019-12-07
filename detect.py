import argparse
from models import Darknet  # set ONNX_EXPORT in models.py
from utils.utils import *
import time


def detect():
    # (320, 192) or (416, 256) or (608, 352) for (height, width)
    img_size, source, weights = opt.img_size, opt.source, opt.weights
    # Initialize
    device = select_device(device=opt.device)

    # Initialize model
    model = Darknet(opt.cfg, img_size)

    # Load weights
    model.load_state_dict(torch.load(weights, map_location=device)['model'])

    # Eval mode
    model.to(device).eval()

    # Set Dataloader
    dataset = load_images(source, img_size=img_size)

    # Get classes and colors
    classes = load_classes("data/coco.names")
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    # Run inference
    t0 = time.time()
    for img, im0s in dataset:
        t = time.time()

        # Get detections
        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.nms_thres)

        # Process detections
        results = []
        det = pred[0]
        s = '%gx%g ' % img.shape[2:]  # print string
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, classes[int(c)])  # add to string

            # Write results
            for *xyxy, conf, _, cls in det:
                result = {"box": [int(xyxy[1]), int(xyxy[3]), int(xyxy[0]), int(xyxy[2])],
                          "confidence": conf.item(),
                          "class": classes[int(cls)]
                          }
                results.append(result)

        print('%sDone. (%.3fs)' % (s, time.time() - t))

        print(results)
    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()
