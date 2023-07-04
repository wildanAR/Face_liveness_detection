import mediapipe as mp
import cv2, time
import numpy as np
import paho.mqtt.client as paho
from keras.models import load_model

broker = "broker.hivemq.com"
port = 1883
no, photos = 0, 0
confidence = 0
def on_message (client, userdata, message):
    msg = str(message.payload.decode("utf-8"))
    t = str(message.topic)
    if (t == "kamera"):
        global no
        no = int(msg)
    return no

client= paho.Client("GUI")
client.on_message=on_message
print("connecting to broker ",broker)
client.connect(broker,port)#connect
print(broker," connected")
client.loop_start()
print("Subscribing to topic")
client.subscribe("kamera")

class FaceDetector():
    """
    model bagus:
    1. lab4
    2. lab10
    3. lab11
    4. lab12
    5. lab13
    6. lab16
    """

    def __init__(self, minDetectionCon=0.5):

        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

        
    def modelCNN(self,img,model = load_model("model/lab16.h5")):
        result = model.predict(img)
        idx = int(result.argmax(axis=1))
        confidence = result.max(axis=1)*100

        return idx, confidence
    
    def findFaces(self, img, draw=True):
        global bbox
 
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bbox = []
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    img = self.fancyDraw(img,bbox)
                    
                    # cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                    #         (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                    #         2, (255, 0, 255), 2)
        return img, bboxs, bbox
 
    def fancyDraw(self, img, bbox, l=30, t=5, rt= 1):
        global x,y,w,h
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        
 
        cv2.rectangle(img, bbox, (0, 255, 0), rt)
        # Top Left  x,y
        cv2.line(img, (x, y), (x + l, y), (0, 255, 0), t)
        cv2.line(img, (x, y), (x, y+l),(0, 255, 0), t)
        # Top Right  x1,y
        cv2.line(img, (x1, y), (x1 - l, y), (0, 255, 0), t)
        cv2.line(img, (x1, y), (x1, y+l), (0, 255, 0), t)
        # Bottom Left  x,y1
        cv2.line(img, (x, y1), (x + l, y1), (0, 255, 0), t)
        cv2.line(img, (x, y1), (x, y1 - l), (0, 255, 0), t)
        # Bottom Right  x1,y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (0, 255, 0), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (0, 255, 0), t)
        return img
    
    def recognition(self,img,kotak = []):
        labels = ['ardi', 'mif' ,'rezqia' ,'wildan']
        global bener, confidence
        try:
            global bener,x,y,w,h
            img, bboxs, bbox = self.findFaces(img)
            x,y,w,h = bbox
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (50, 50))
            face_img = face_img.reshape(1,50,50,1)
            face_img = np.array(face_img, dtype=np.float32)
            idx, confidence = self.modelCNN(face_img)
            nama = labels[idx]


            if len(kotak) != 50:
                kotak.append(nama)
                bener = max(set(kotak), key=kotak.count)
            else: del kotak[:]


            if confidence >= 80:
                cv2.putText(img, f'{str(bener)}',
                            (x, (y + h) +30), cv2.FONT_HERSHEY_PLAIN,
                            2, (0, 255, 0), 2)
                cv2.putText(img, f'{int(confidence)} %',
                            (x, (y + h) -200), cv2.FONT_HERSHEY_PLAIN,
                            2, (0, 255, 0), 2)
                
            else:
                cv2.putText(img, 'tidak dikenali',   
                            (x, (y + h) +30), cv2.FONT_HERSHEY_PLAIN,
                            2, (0, 255, 0), 2)

        except:
            if confidence >= 80:
                cv2.putText(img, f'{str(bener)}',
                            (x, (y + h) +30), cv2.FONT_HERSHEY_PLAIN,
                            2, (0, 255, 0), 2)
                cv2.putText(img, f'{int(confidence)} %',
                            (x, (y + h) -200), cv2.FONT_HERSHEY_PLAIN,
                            2, (0, 255, 0), 2)
                
            else:
                # print(x)
                cv2.putText(img, 'tidak dikenali',   
                            (x, (y + h) +30), cv2.FONT_HERSHEY_PLAIN,
                            2, (0, 255, 0), 2)
        return img
    
    def testPredict(self, kotak = []):
        cam = cv2.VideoCapture('video/sean_vid.mp4')
        # cam = cv2.VideoCapture(0)
        labels = ['ardi', 'mif' ,'rezqia' ,'wildan']
        global confidence
        
        while True:
            ret, frame = cam.read()
            # frame = cv2.imread('foto/tes4.jpg')
            global bener, confidence
            try:
                global bener, confidence
                global bener,x,y,w,h
                img, bboxs, bbox = self.findFaces(img)
                x,y,w,h = bbox
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (50, 50))
                face_img = face_img.reshape(1,50,50,1)
                face_img = np.array(face_img, dtype=np.float32)
                idx, confidence = self.modelCNN(face_img)
                nama = labels[idx]

                if len(kotak) != 50:
                    kotak.append(nama)
                    bener = max(set(kotak), key=kotak.count)
                else: del kotak[:]

                print(confidence)

                if confidence >= 80:
                    cv2.putText(img, f'{str(bener)}',
                                (x, (y + h) +30), cv2.FONT_HERSHEY_PLAIN,
                                2, (0, 255, 0), 2)
                    cv2.putText(img, f'{int(confidence)} %',
                                (x, (y + h) -200), cv2.FONT_HERSHEY_PLAIN,
                                2, (0, 255, 0), 2)
                    
                else:
                    cv2.putText(img, 'tidak dikenali',   
                                (x, (y + h) +30), cv2.FONT_HERSHEY_PLAIN,
                                2, (0, 255, 0), 2)

            except:
                
                if confidence >= 80:
                    cv2.putText(img, f'{str(bener)}',
                                (x, (y + h) +30), cv2.FONT_HERSHEY_PLAIN,
                                2, (0, 255, 0), 2)
                    cv2.putText(img, f'{int(confidence)} %',
                                (x, (y + h) -200), cv2.FONT_HERSHEY_PLAIN,
                                2, (0, 255, 0), 2)
                    
                else:
                    cv2.putText(img, 'tidak dikenali',   
                                (x, (y + h) +30), cv2.FONT_HERSHEY_PLAIN,
                                2, (0, 255, 0), 2)
            img = cv2.resize(img, (500,500))
            cv2.imshow('cek',img)
            cv2.waitKey(10)




def main():
    time_now = 0
    time_prev = time.time()
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()
    while True:
        time_now = time.time() - time_prev
        _, img = cap.read()
        img = cv2.imread("foto/2_2.jpg")
        detector.recognition(img)
        cv2.imshow("Image", img)
        cv2.waitKey(100)
        if no == 0:
            if (time_now > 0.01):
                try:
                    photos += 1
                    cv2.imwrite(f"D:/Non_Akademik/cek/1_{photos}.jpg", img)

                    cv2.destroyAllWindows()
                    cv2.waitKey()
                except:
                    pass
 

if __name__ == "__main__":
    # FaceDetector().testPredict() #rusak
    main()
# FaceDetector()