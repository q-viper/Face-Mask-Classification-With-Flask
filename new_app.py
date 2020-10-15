from flask import Flask, render_template, Response, make_response,jsonify
import cv2
import numpy as np
import matplotlib.pyplot as plt
from new_main import *
import time
import os
from tensorflow.keras.models import load_model, clone_model
import threading

lock = threading.Lock()

dpath=dir_path = os.path.dirname(os.path.realpath(__file__))
err_img=cv2.imread(dpath+"\static\error_frame.png")
err_img = np.ones((100, 100, 3), dtype=np.uint8)
_, err_img_byte = cv2.imencode(".jpg", err_img)
err_img_byte = err_img_byte.tobytes()

def model_loader(model_dir):
        loaded_model = load_model(model_dir+".h5")
        print("Loaded model from disk")
        return loaded_model
model=model_loader(model_dir="F:\Desktop\cv event\cvevent-faceapp\models\customCNN64")

class FaceRecWeb:
    def __init__(self, model):
        self.camera = FaceDet(model)
        self.count=0
        self.mask="Detecting"
        
    def frame_gen(self, camera, kind="frame"):        
        while True:    
            camera.main()
            threading.Thread(target=camera.mask_classifier(model=clone_model(model)))
            frame=camera.get_frames()
            if frame is None:
                frame = err_img_byte
            else:
                count=camera.count
                mask=camera.res
                gw.count = count
                gw.mask = mask
            frame = (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'+bytearray(frame)+b'\r\n')
            if kind =="frame":
                yield frame
              
  
app = Flask(__name__)

@app.before_first_request
def before_first_request_func():
    global gw, count, mask
    count = 0
    mask="Detecting" 
    gw = FaceRecWeb(model)
    

@app.route('/')
def index():    
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    fresp = gw.frame_gen(gw.camera, kind="frame")
    return Response(fresp, mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_count')
def get_count():
    return Response(str(gw.count), mimetype='text')

@app.route('/get_mask')
def get_mask():
    return Response(str(gw.mask), mimetype='text')

 
@app.teardown_request
def teardown_request_func(error=None):
    global gw
    gw = FaceRecWeb(model)   


if __name__=='__main__':  
    app.run(debug=True, threaded=True)
    
    