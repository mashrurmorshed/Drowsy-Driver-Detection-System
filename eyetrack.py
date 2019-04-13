# python eyetrack.py -p shape_predictor_68_face_landmarks.dat

from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2

#command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=str, default="",
	help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0,
	help="index of webcam on system")
args = vars(ap.parse_args())


def sound_alarm(path):
	playsound.playsound(path)

def eye_aspect_ratio(eye,f,gray):
        #vertical(1), vertical(2)
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
        #horizontal
	C = dist.euclidean(eye[0], eye[3])

	# aspect ratio= vertical(1)+vertical(2)/(2*horizontal)
	ear = (A + B) / (2.0 * C)
	
	r1=0
	c1=0
	r2=0
	c2=0
	
	if ear>.2:
                l=eye[0]
                r=eye[3]
                m=(r[1]-l[1])/(r[0]-l[0])
                print(l,' ',r)
                print('slope ',m)
                prev=l[1]
                
                previous_pixel=200;
                for i in range(l[0]+3,r[0]-3):
                        j=round(m+prev)
                        print(gray[int(j)][int(i)])
                        prev=j
                        current_pixel=gray[int(j)][int(i)]
                        if current_pixel<100 and previous_pixel>=100:
                                r1=j
                                c1=i
                        if current_pixel>=100 and previous_pixel<100:
                                r2=j
                                c2=i
                                break
                        if i==r[0]-4:
                                r2=j
                                c2=i
                        previous_pixel=current_pixel
	if c2==0 or c1==0:
                radius=0
	else:
                radius=abs(c2-c1)/2
	mr=(r1+r2)/2
	mc=(c1+c2)/2
	print(c1,' ',c2,' ',r1,' ',r2,' ',radius,' ',mr,' ',mc)
	return (ear,radius,mr,mc)
        

 

#AR < threshold triggers alarm
EYE_AR_THRESH = 0.3
#Number of consecutive frames to check
EYE_AR_CONSEC_FRAMES = 50

COUNTER = 0
ALARM_ON = False

# initialize dlib's HOG face detector
print("Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

#start and end of left eye and right eye(out of 62 landmarks, siz given to each eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

# loop over frames from the video stream
f=0
while True:
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	frame = imutils.resize(frame, width=450)

	#potential quality improvement
	#frame2=frame
	#b,g,r = cv2.split(frame)
	#b = cv2.equalizeHist(b)
	#g = cv2.equalizeHist(g)
	#r = cv2.equalizeHist(r)
	#frame = cv2.merge((b,g,r))
        
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	if f==0:
                print(np.shape(gray))
                #print(type(gray))
                

                
	

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR,lr,cr1,cc1 = eye_aspect_ratio(leftEye,f,gray)
		rightEAR,rr,cr2,cc2 = eye_aspect_ratio(rightEye,f,gray)
		
		if lr>0:
			cv2.circle(frame,(int(cc1),int(cr1)),int(lr),(0,0,255), -1)
		if rr>0:
			cv2.circle(frame,(int(cc2),int(cr2)),int(rr),(0,0,255), -1)
		if f==0:
                        f=1

		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if ear < EYE_AR_THRESH:
			COUNTER += 1

			# if the eyes were closed for a sufficient number of
			# then sound the alarm
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				# if the alarm is not on, turn it on
				if not ALARM_ON:
					ALARM_ON = True

					# check to see if an alarm file was supplied,
					# and if so, start a thread to have the alarm
					# sound played in the background
					if args["alarm"] != "":
						t = Thread(target=sound_alarm,
							args=(args["alarm"],))
						t.deamon = True
						t.start()

				# draw an alarm on the frame
				cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# otherwise, the eye aspect ratio is not below the blink
		# threshold, so reset the counter and alarm
		else:
			COUNTER = 0
			ALARM_ON = False

		# draw the computed eye aspect ratio on the frame to help
		# with debugging and setting the correct eye aspect ratio
		# thresholds and frame counters
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	# show the frame
        #frame=cv2.flip(frame,1,0)  np.hstack((frame,frame2))
	
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
