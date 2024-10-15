#############################################
# Object detection via RTSP - YOLO - OpenCV
# Author : Myth
############################################
import logging
import os.path
# pip install opencv-python
import cv2
import argparse
import numpy as np
# import imageio_ffmpeg as imageio
from ultralytics import YOLO
import subprocess
from deepface import DeepFace
from stream_openai_video import video_produce

logging.basicConfig(level=logging.INFO)

# 创建一个ArgumentParser对象，用于处理命令行参数
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=False,
                help='path to input image', default='rtp://0.0.0.0:19292')
ap.add_argument('-o', '--outputfile', required=False,
                help='filename for output video', default='output.mp4')
ap.add_argument('-od', '--outputdir', required=False,
                help='path to output folder', default='output')
ap.add_argument('-fs', '--framestart', required=False,
                help='start frame', default=0)
ap.add_argument('-fl', '--framelimit', required=False,
                help='number of frames to process (0 = all)', default=0)
ap.add_argument('-ic', '--invertcolor', required=False,
                help='invert RGB 2 BGR', default='False')
ap.add_argument('-fpt', '--fpsthrottle', required=False,
                help='skips (int) x frames in order to catch up with the stream for slow machines 1 = no throttle',
                default=10)
# 解析命令行参数并存储在args对象中
args = ap.parse_args()

# Load a pre-trained YOLOv10n model (this should be done once, not inside the detect function if called frequently)
model = YOLO("yolo/config/yolov10x.pt")
input_stream_config = "yolo/config/stream.sdp"

command = [
    'ffmpeg',
    '-protocol_whitelist', 'file,udp,rtp',
    '-i', input_stream_config,
    '-f', 'image2pipe',
    '-pix_fmt', 'bgr24',
    '-vcodec', 'rawvideo', '-'
]

names_array = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


# 将字符串转换为布尔值的辅助函数
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# 获取YOLO模型输出层的名称
def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


# 检测函数，使用YOLO模型进行对象检测
def detect(image):
    # Perform object detection on the input image
    results = model(image)

    # Extracting detections
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding box coordinates
    confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # Class IDs

    # var = results[0].show
    # logging.info(f'detection results: {var}')

    # Optionally: perform NMS or other post-processing (if needed)
    # The ultralytics library already performs NMS by default, so this may not be necessary.
    # 根据是否包含人脸决定是否调用人像识别函数
    containsPerson = True
    # 初始化一个字典来存储物体名称及其数量
    detected_objects = {}

    # orgImage = image.copy()
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        confidence = confidences[i]
        class_id = class_ids[i]
        object_name = names_array[class_id]

        logging.info(
            f"Detected object: Class ID = {class_id},"
            f"Name = {object_name},"
            f" Confidence = {confidence:.2f}, "
            f"Bounding Box = [{round(x1)}, {round(y1)}, {round(x2)}, {round(y2)}]")

        # 更新检测物体的数量
        if object_name in detected_objects:
            detected_objects[object_name] += 1
        else:
            detected_objects[object_name] = 1

        if class_id == 0:
            containsPerson = True

    # 将检测到的物体及其数量汇总为文本
    summary_text = "Detected objects summary:\n"
    for object_name, count in detected_objects.items():
        summary_text += f"- {object_name}: {count}\n"

    # 将检测的物体汇总发送到队列
    video_produce(summary_text)

    if containsPerson:
        objs = DeepFace.analyze(
            image,
            actions=['age', 'gender', 'race', 'emotion'],
            enforce_detection=False
        )
        person_text = "Detected person summary:\n"
        for person in objs:
            logging.info(f"Age: {person['age']}")
            logging.info(f"Gender: {person['gender']}")
            logging.info(f"Race: {person['race']}")
            logging.info(f"Emotion: {person['emotion']}")
            person_text += f"- {person['age']}: {person['gender']} - {person['race']}: {person['emotion']}\n"
            logging.info("------------------------")
        # 将检测的人像汇总发送到队列
        video_produce(person_text)
    return image


# 处理视频文件
def processvideo(file):
    cap = cv2.VideoCapture(file)
    frame_counter = 0
    while (cap.isOpened()):
        frame_counter = frame_counter + 1
        ret, frame = cap.read()
        logging.info('Detecting objects in frame ' + str(frame_counter))
        if ret == True:
            if not frame is None:
                detect(frame)
            else:
                logging.info('Frame error in frame ' + str(frame_counter))
        else:
            break
    cap.release()


width = 1920
height = 1080


class YoloOpencvProcessor:
    def __init__(self):
        self.frame_counter = 0  # 每个会话独立的计数器

    def process_frame(self, frame):
        if frame is not None:
            if int(args.framelimit) > 0 and self.frame_counter > int(args.framestart) + int(args.framelimit):
                return
            if self.frame_counter % int(args.fpsthrottle) == 0:
                image = frame.to_ndarray(format="bgr24")
                if len(image) == 0:
                    logging.info('Frame error in frame is null!')
                    return
                detect(image)
                logging.info(f'Detecting objects in frame {self.frame_counter}')

            # 每次调用时计数器增加
            self.frame_counter += 1


def yolo_opencv_main():
    # 主函数逻辑，根据输入类型选择处理方式
    if args.input.startswith('rtp'):
        logging.info('Starting RTP video capture')
        pipe = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10 ** 2)
        counter = 0
        while (True):
            raw_image = pipe.stdout.read(width * height * 3)
            if len(raw_image) == 0:
                logging.info('Frame error in rtp ' + str(counter))
                break
            if int(args.framelimit) > 0 and counter > int(args.framestart) + int(args.framelimit):
                break

            if counter % int(args.fpsthrottle) == 0:
                image = np.frombuffer(raw_image, dtype='uint8').reshape((height, width, 3))
                detect(image)
                logging.info('Detecting objects in rtp ' + str(counter))
            # else:
            #     logging.info('FPS throttling. Skipping frame ' + str(frame_counter))
            counter = counter + 1
    else:
        if os.path.isdir(args.input):
            for dirpath, dirnames, filenames in os.walk(args.input):
                for filename in [f for f in filenames if f.endswith(".mp4")]:
                    logging.info('Processing video ' + os.path.join(dirpath, filename))
                    processvideo(os.path.join(dirpath, filename))
        else:
            processvideo(os.path.join(args.input))
