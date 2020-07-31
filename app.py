from flask import Flask, Response, render_template
import cv2
import numpy as np
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils_huy as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import base64

FRAMEWORK = 'tf'
WEIGHTS = './checkpoints/yolov4-tiny-416'
SIZE = 416
TINY = True
MODEL = 'yolov4'
# flags.DEFINE_string('video', './data/test_vid.mp4', 'path to input video')
IOU = 0.45
SCORE = 0.25

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config("any thing")
    input_size = SIZE
    video_path = 'rtsp://anhdn:123456@172.10.1.17:554'
    # video_path = FLAGS.video


    #out = cv2.VideoWriter('result.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (720,480))
    #vid.set(cv2.CAP_PROP_FPS, )

    if FRAMEWORK == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=WEIGHTS)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    else:
        saved_model_loaded = tf.saved_model.load(WEIGHTS, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    print("Video from: ", video_path )
    vid = cv2.VideoCapture(video_path)
    return_value, frame = vid.read()

    while True:
        prev_time = time.time()

        # return_value, frame = vid.read()
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
            #print(image.shape)

        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        if FRAMEWORK == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if MODEL == 'yolov4' and TINY == True:
                print('ssss')
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25)
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25)
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=IOU,
            score_threshold=SCORE
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        image = utils.draw_bbox(frame, pred_bbox)
        curr_time = time.time()
        exec_time = curr_time - prev_time
        norm_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        result = np.asarray(norm_image)

        #ret, frame = cap.read()
        #image = cv2.imencode('.jpg', frame)[1].tobytes()
        image_encode = cv2.imencode('.jpeg', result)[1].tobytes()
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + image_encode + b'\r\n')
        #time.sleep(0.1)
        #return_value, frame = vid.read()
        #end = time.time()
        #print('duration per frame: ')9
        #print(round((exec_time) * 1000))
    #cap.release()
    #return frame

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='10.11.6.13', threaded=True)