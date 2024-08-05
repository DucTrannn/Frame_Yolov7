import cv2
import torch
import numpy as np
import time  
import os
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.torch_utils import select_device

def detect():
    log_file_path = 'FrameYolov7.log'

    if os.path.exists(log_file_path):
        os.remove(log_file_path)

    device = select_device('0')

    if device.type == 'cuda':
        print("Dung GPU")
    else:
        print("Dung CPU")

    model = attempt_load('yolov7.pt', map_location=device)
    model.eval()

    video_path = '/home/ubuntu/yolov7/video/yolo.mp4'
    cap = cv2.VideoCapture(video_path)
    log_file = open(log_file_path, 'w')

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    prev_time = time.time()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        img = letterbox(frame, new_shape=640)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float().div(255.0).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(img)[0]
            pred = non_max_suppression(pred, 0.25)

        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in det:
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, frame, label=label)

        end_time = time.time()
        fps = 1.0 / (end_time - start_time)
        current_video_time = frame_count / video_fps
        frame_count += 1
        print(f'FPS:  {fps:.2f},  Time: {current_video_time:.2f}s')
        log_file.write(f'FPS: {fps:.2f},  Time: {current_video_time:.2f}s\n')
        cv2.imshow('YOLOv7 ', frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    log_file.close()

def plot_one_box(xyxy, img, color=(0, 255, 0), label=None, line_thickness=3):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)
    if label:
        tf = max(line_thickness - 1, 1)
        t_size = cv2.getTextSize(label, 0, line_thickness / 3, tf)[0]
        c1, c2 = x1, y1
        c1 = (c1, c2 - t_size[1] - 3)
        c2 = (c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4)
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (x1, y1 - 2), 0, line_thickness / 3, [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

if __name__ == "__main__":
    detect()
