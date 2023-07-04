from face_recognition.Face_detect import FaceDetector
from liveness_detection.liveness_detector import question_bank,detect_liveness, challenge_result,arah_wajah
import cv2 as cv
import random, imutils
import numpy as np

def show_image(cam,text,color = (0,0,255)):
    ret, im = cam.read()
    im = imutils.resize(im, width=720)
    im = cv.flip(im, 1)
    cv.putText(im,text,(10,50),cv.FONT_HERSHEY_COMPLEX,1,color,2)
    return im

COUNTER, TOTAL = 0,0
counter_ok_questions = 0
counter_ok_consecutives = 0
limit_consecutives = 4
limit_questions = 4
counter_try = 0
limit_try = 50 
detector = FaceDetector()

# lab
labels = ['ardi', 'mif' ,'rezqia' ,'wildan']
cam = cv.VideoCapture(0)
person = 0



for i_questions in range(0,limit_questions):
# genero aleatoriamente pregunta
    index_question = random.randint(0,1)
    question = question_bank(index_question)
    
    im = show_image(cam,question)
    # im = recog(im)
    cv.imshow('liveness_detection',im)
    if cv.waitKey(1) &0xFF == ord('q'):
        break 

    for i_try in range(limit_try):
        # <----------------------- ingestar data 
        ret, im = cam.read()
        im = imutils.resize(im, width=720)
        im = cv.flip(im, 1)
        # <----------------------- ingestar data 
        TOTAL_0 = TOTAL
        out_model,leftEye = detect_liveness(im,COUNTER,TOTAL_0)
        TOTAL = out_model['total_blinks']
        COUNTER = out_model['count_blinks_consecutives']
        dif_blink = TOTAL-TOTAL_0
        if dif_blink > 0:
            blinks_up = 1
        else:
            blinks_up = 0

        challenge_res = challenge_result(question, out_model,blinks_up)

        im = show_image(cam,question)
        cv.imshow('liveness_detection',im)
        if cv.waitKey(1) &0xFF == ord('q'):
            break 

        if challenge_res == "pass":
            im = show_image(cam,question+" : ok")
            cv.imshow('liveness_detection',im)
            if cv.waitKey(1) &0xFF == ord('q'):
                break

            counter_ok_consecutives += 1
            if counter_ok_consecutives == limit_consecutives:
                counter_ok_questions += 1
                counter_try = 0
                counter_ok_consecutives = 0
                break
            else:
                continue

        elif challenge_res == "fail":
            counter_try += 1
            show_image(cam,question+" : fail")
        elif i_try == limit_try-1:
            break
            

    if counter_ok_questions ==  limit_questions:
        global tes1
        tes1 = False

        while True:
            arah = arah_wajah(im)
            out,leftEye = detect_liveness(im,COUNTER,TOTAL)
            # tags = (out['emotion']+out['orientation'])
            # tags = str(tags)
            # im = recog(cam,tags)
            ret, frame = cam.read()
            # frame = cv.imread('foto/tes4.jpg')
            im = detector.recognition(frame)
            cv.circle(im,leftEye[0],radius=0,color=(0,0,255),thickness=5)
            cv.circle(im,leftEye[1],radius=0,color=(0,0,255),thickness=5)
            cv.circle(im,leftEye[2],radius=0,color=(0,0,255),thickness=5)
            cv.circle(im,leftEye[3],radius=0,color=(0,0,255),thickness=5)
            cv.circle(im,leftEye[4],radius=0,color=(0,0,255),thickness=5)
            cv.circle(im,leftEye[5],radius=0,color=(0,0,255),thickness=5)
            cv.imshow("cek",im)
            if cv.waitKey(1) &0xFF == ord('q'):
                break
    elif i_try == limit_try-1:
        while True:
            im = show_image(cam,"LIFENESS FAIL")
            cv.imshow('liveness_detection',im)
            if cv.waitKey(1) &0xFF == ord('q'):
                break
        break 

    else:
        continue