import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
from streamlit_webrtc import WebRtcMode, webrtc_streamer, VideoTransformerBase
import av

class_name = ["anpanman","baikinman","rollpanna"]

@st.cache_resource
def load_model():
    model_path = r'saved_model'
    return tf.saved_model.load(model_path)

model = load_model()

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    # 解像度のダウンスケール
    #img = cv2.resize(img, (960, 540))
    result = detect_objects(img)

    # ダウンスケール後の画像を元の解像度に戻す
    #result = cv2.resize(result, (frame.width, frame.height))

    return av.VideoFrame.from_ndarray(result, format="bgr24")

def detect_objects(img):
    # 画像の前処理
    input_tensor = tf.convert_to_tensor(img)
    input_tensor = input_tensor[tf.newaxis, ...]
    
    detections = model.signatures['serving_default'](input_tensor)

    # Get bounding boxes, scores, and classes from detections
    boxes = detections['detection_boxes'][0]
    scores = detections['detection_scores'][0]
    classes = detections['detection_classes'][0]

    # 物体検出結果の表示
    for box, score, cls in zip(boxes, scores, classes):
        # Only display boxes with scores greater than 0.5
        if int(cls) == 2: #バイキンマンのみ
            if score > 0.9:
                # Scale box coordinates to image size
                height, width, _ = img.shape
                y1, x1, y2, x2 = box
                y1 = int(y1 * height)
                x1 = int(x1 * width)
                y2 = int(y2 * height)
                x2 = int(x2 * width)
                img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Optionally, you can display class label and score.
                label = f"{class_name[int(cls)-1]}, Score: {score:.2f}"
                img = cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img

st.title("推しDetector(バイキンマンver)")
webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    rtc_configuration={ "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": True},
    async_processing=True,
)
