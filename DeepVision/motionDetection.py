import time
import cv2
import imutils

from Logger import log

#Loop
camera_port = 0
camera = cv2.VideoCapture(camera_port)

min_area = 500

imgCounter = 0

#Code inspiration
#https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/

def motionDetected(previousFrame,currentFrame):
	pFrame = imageTransform(previousFrame)
	cFrame = imageTransform(currentFrame)

	frameDelta = cv2.absdiff(pFrame, cFrame)
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.dilate(thresh, None, iterations=2)
	(_,cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	significantMovement = False

	for c in cnts:
		# if the contour is too small, ignore it
		if(cv2.contourArea(c) < min_area):
			continue
		significantMovement=True
 	    # compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(currentFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		text = "Intrusion Detected"

		# draw the text and timestamp on the frame
		import datetime
		cv2.putText(currentFrame, "Room Status: {}".format(text), (10, 20),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		cv2.putText(currentFrame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
			(10, currentFrame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

	return significantMovement,currentFrame

def imageTransform(image):
	frame = imutils.resize(image, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)
 
	return (gray)

def storeSnapshot(frame,counter=0):
	import datetime
	datetime_suffix = datetime.datetime.now().strftime("%B-%d-%Y %H-%M-%S%p")
	
	filename = "watchdog/outdoors_{0}_ShotCount{1}{2}".format(datetime_suffix,counter,".jpg")
	cv2.imwrite(filename, frame)
	return




def snapShotIfMotion(previousFrame):
	(grabbed, currentFrame) = camera.read()
	cpyFrame = currentFrame.copy()

	if previousFrame is None:
		previousFrame = currentFrame

	if  grabbed:
		motion,processedframe = motionDetected(previousFrame,cpyFrame)
		if(motion):
			#Capture the frame
			log.info("Motion detected")
			storeSnapshot(processedframe)
			mDetect = True
		else:
			mDetect = False
		
		#previousFrame = currentFrame
		return mDetect,currentFrame

	else:
		log.debug("Camera capture failed")
		return False,currentFrame

def snapShot(NrOfTimes):
	currentFrame = None
	for i in range(1,NrOfTimes):
		(grabbed, currentFrame) = camera.read()
		if(grabbed):
			storeSnapshot(currentFrame,i)
		else:
			log.debug("Camera capture failed")

	return currentFrame







	
	

