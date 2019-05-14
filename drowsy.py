# USAGE
# python drowsy.py 

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import imutils
import time
import dlib
import cv2

def sound_alarm(path):
	# play an alarm sound
	playsound.playsound(path)

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear

def eyeinfo(eye,f):
	info= "\""+str(eye[0][0])+", "+str(eye[0][1])+"\",\""+str(eye[3][0])+", "+str(eye[1][1])+"\""
	if f==1:
		return "Left Eye,"+info
	else:
		return "Right Eye,"+info


alarmfile="alarm.wav"
shape_pred="shape_predictor_68_face_landmarks.dat"
wcam=0 #d
 
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 32
BL_THRESH = 3

# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0
BLCOUNTER = 0
BLINKS= 0
LASTBL=0
ALARM_ON = False

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_pred)

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=wcam).start()
time.sleep(1.0)


file1 = open("log.csv","a")
file1.write("Time, Event, From, To, Blinkcount,Frame rate\n")

start= time.time()
start_time = time.asctime(time.localtime(time.time()))
file1.write(str(start_time)+", Start\n")

framecount=0
wcount=0
f=0
while True:
	framecount+=1
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	
	#b,g,r = cv2.split(frame)
	
	#frame = cv2.merge((b,g,r))

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#gray = cv2.equalizeHist(gray)

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
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		if framecount%90==0:
			
			tm = str(time.asctime(time.localtime(time.time())))
			Linfo=eyeinfo(leftEye,1)
			Rinfo=eyeinfo(rightEye,2)
			file1.write(tm+","+Linfo+"\n")
			file1.write(tm+","+Rinfo+"\n")

		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0

		cv2.putText(frame, "Eye Ratio: {:.2f}".format(ear), (260, 30),
			cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
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
			BLCOUNTER +=1
			# if the eyes were closed for a sufficient number of
			# then sound the alarm
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				# if the alarm is not on, turn it on
				if not ALARM_ON:
					ALARM_ON = True

					# check to see if an alarm file was supplied,
					# and if so, start a thread to have the alarm
					# sound played in the background
					
					t = Thread(target=sound_alarm,
						args=(alarmfile,))
					t.deamon = True
					t.start()

				
				cv2.putText(frame, "WARNING!!!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
				if f==0:
					wtime = time.asctime(time.localtime(time.time()))
					file1.write(str(wtime)+",Drowsiness\n")
					f=1
				wcount +=1
				if (wcount>50):
					wcount=0
					f=0



		# otherwise, the eye aspect ratio is not below the blink
		# threshold, so reset the counter and alarm
		else:
			if BLCOUNTER >= BL_THRESH:
				BLINKS += 1
			BLCOUNTER = 0
			COUNTER = 0
			ALARM_ON = False
			f=0

		# draw the computed eye aspect ratio on the frame to help
		# with debugging and setting the correct eye aspect ratio
		# thresholds and frame counters

	cv2.putText(frame, "Blinks: "+str(BLINKS), (260, 70),
					cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

	if framecount%300==0:
		avbl=int((BLINKS-LASTBL))
		LASTBL=BLINKS
		file1.write(" ,Blinks/10s, , ,"+str(avbl)+"\n")



	curr= round(time.time()-start)
	secs= int(curr%60)
	curr= curr-secs
	mins= int((curr/60)%60)
	hrs= int(((curr/60)-mins)/60)
	cv2.putText(frame, str(hrs)+":"+str(mins)+":"+str(secs), (330, 300),
					cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
	

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
end=time.time()
totalruntime=end-start
fps=round(framecount/totalruntime)


end_time = time.asctime(time.localtime(time.time()))
file1.write(str(end_time)+",End\n")
file1.write(" ,frame/sec, , , ,"+str(fps)+"fps\n")
file1.write("-,-,-,-,-,-\n\n")
file1.close()


cv2.destroyAllWindows()
vs.stop()
