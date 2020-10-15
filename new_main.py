import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import imutils
import time
from tensorflow.keras.models import load_model, clone_model
import time
from multiprocessing.pool import ThreadPool
from collections import deque
import threading

view=False

def model_loader(model_dir):
        loaded_model = load_model(model_dir+".h5")
        print("Loaded model from disk")
        return loaded_model
model=model_loader(model_dir="F:\Desktop\cv event\cvevent-faceapp\models\customCNN64")


dpath=dir_path = os.path.dirname(os.path.realpath(__file__))

class FaceDet:
    def __init__(self, model):
        self.num_faces=0
        self.current_frame = None
        self.cascade_dir = dpath + "\opencv-files"
        #print((self.cascade_dir+'\haarcascade_frontalface_default.xml'))
        self.face_cascade = cv2.CascadeClassifier(self.cascade_dir+'\haarcascade_frontalface_default.xml')
        self.cam = cv2.VideoCapture(0)
        
        # self.loaded_model = model_loader(model_dir)
        self.loaded_model=model
        self.model_shape=self.loaded_model.input_shape[1:3]
        self.face_img=np.zeros((self.model_shape[0], self.model_shape[1], 3))
        self.res, self.count, self.acc = "Detecting", 0, 0
        # self.thread = threading.Thread(target=self.mask_classifier(), args=())
        # self.thread.daemon = True
        # self.thread.start()
        
    
    def mask_classifier(self, image=None, model=None):
        results = []
        if model is not None:
            self.loaded_model=model
        #print(len(self.faces))
        if len(self.faces) > 0:
            for (x,y,w,h) in self.faces:
                image=self.rgb_clone[y:y + h, x:x + w]
                img = cv2.resize(image, self.model_shape)
                img = img.reshape(1, self.model_shape[0], self.model_shape[1], 3)/255
                acc = self.loaded_model.predict(img)
                prediction = np.argmax(acc)
                classes = ["Mask", "Unmasked"]
                res = classes[prediction]
                self.res, self.acc = res, round(acc[0][prediction], 4)
                results.append((x,y,w,h, res, self.acc))
        self.faces=results
        #return res, round(acc[0][prediction], 4)

    def __del__(self):
        self.cam.release()
        #cv2.destroyAllWindows()
    
    def get_frames(self):
        fres=""
        if len(self.faces) > 0:
            i=0
            for (x,y,w,h,res,acc) in self.faces:
                color = (0, 255, 0)
                clone = self.rgb_clone
                if "Unmasked" in res.split(" "):
                    color=(0, 0, 255)
                res += " "+str(acc)
                res = f"Person: {i} "+res
                cv2.putText(clone, res, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                cv2.rectangle(clone,  (x, y), (x + w, y + h), color, 2)
                i+=1
                fres+=res+"\n"
            clone=cv2.cvtColor(clone, cv2.COLOR_BGR2RGB)
            ret, clone_jpeg = cv2.imencode('.jpeg', clone)
            if ret:
                self.clone = clone_jpeg.tobytes()
                if view:
                    cv2.imshow("frame", clone)
                else:
                    self.count=len(self.faces)
                    self.res=fres
                    return clone_jpeg.tobytes()
        
        
    def main(self):
        while True:
            ret, frame = self.cam.read()
            if ret:
                self.key = cv2.waitKey(1) & 0xFF
                frame = cv2.flip(frame, 1)
                clone = frame.copy()
                gray = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)
                self.rgb_clone = cv2.cvtColor(clone, cv2.COLOR_BGR2RGB)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
                d=20
                shape = gray.shape
                res=""  
                self.faces=[] 
                rgb_clone=self.rgb_clone         
                for i in range(0, len(faces)):
                    (x, y, w, h) = faces[i]
                    y = np.clip(y-d, 0, y)
                    x = np.clip(x-d, 0, x)
                    w = np.clip(w+2*d, w, shape[0]-w-2*d)
                    h = np.clip(h+2*d, h, shape[1]-h-2*d)
                    # pass this to classifier
                    # face = gray[y:y + h, x:x + w] 
                    # img = rgb_clone[y:y + h, x:x + w]
                    # print("x,y")
                    # self.face_img = img
                    # # res, acc = self.mask_classifier()
                    # res, acc=self.res, self.acc
                    
                    # color = (0, 255, 0)
                    # if res == "Unmasked":
                    #     color=(0, 0, 255)
                    # res += " "+str(acc)
                    # res = f"Person: {i} "+res
                    # cv2.putText(clone, res, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                    # cv2.rectangle(clone,  (x, y), (x + w, y + h), color, 2)
                    self.faces.append((x, y, w, h))
                threading.Thread(target=self.mask_classifier(model=clone_model(model)))
                if view:
                    self.mask_classifier()
                    self.get_frames()
                else:
                    return None
                # ret, clone_jpeg = cv2.imencode('.jpeg', clone)
                # if ret:
                #     self.clone = clone_jpeg.tobytes()
                #     # cv2.imshow("frame", clone)
                #     self.count=len(faces)
                #     self.res=res
                #     return clone_jpeg.tobytes(), len(faces), res
                
    


if __name__ == "__main__":
    # pass
    view=True
    fd = FaceDet(model)
    fd.main()
    