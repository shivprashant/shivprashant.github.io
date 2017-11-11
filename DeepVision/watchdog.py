import time
import cv2
import imutils

from Logger import log

class WatchDog:

	def __init__(self,camera,intruderLDM):
		self.eyes = camera
		self.mDetected = False
		self.nrofInstrusions = 0
		
		#Nr of frame to be captured continuously in even of intrusion
		self.maxFrameCaptureLimit = 20

		#Min area threshhold for motion detection
		self.min_area = 500

		#path of current snapshot post motion is detected.
		#Initialized when snapshot is taken. Used as input to intruderLDM
		suspectedIntrustion = None


	def __recordIntrusion(self):
		return True

	def motionDetected(self,previousFrame,currentFrame):
		pFrame = self.imageTransform(previousFrame)
		cFrame = self.imageTransform(currentFrame)

		frameDelta = cv2.absdiff(pFrame, cFrame)
		thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
		thresh = cv2.dilate(thresh, None, iterations=2)
		(_,cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

		significantMovement = False

		for c in cnts:
			# if the contour is too small, ignore it
			if(cv2.contourArea(c) < self.min_area):
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

	def storeSnapshot(self,frame,counter=0):
		import datetime
		datetime_suffix = datetime.datetime.now().strftime("%B-%d-%Y %H-%M-%S%p")
		
		filename = "watchdog/outdoors_{0}_ShotCount{1}{2}".format(datetime_suffix,counter,".jpg")
		cv2.imwrite(filename, frame)
		return filename

	def imageTransform(self,image):
		frame = imutils.resize(image, width=500)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (21, 21), 0)
	 
		return (gray)

	def __recordIntrusion(self):
		for i in range(1,self.maxFrameCaptureLimit):
			(grabbed, cFrame) = self.eyes.fetchCurrent()
			if(grabbed):
				storeSnapshot(cFrame,i)
			else:
				log.error("Camera capture failed")
				return False

		return True

	def watch(self):
		
		while(1):
			if(self.mDetected == True):
				#Intrusion is detected. 
				if(nrofInstrusions>0):
					log.info("Logging intrustion : ",nrofInstrusions)
					if(intruderLDM.isFamily(suspectedIntrustion) == False):
						__recordIntrusion(self)

				nrofInstrusions = nrofInstrusions + 1

				self.mDetected = False
				self.suspectedIntrustion = None

			else:
				#Capture and Analyze frame
				_,cFrame = self.eyes.fetchCurrent()
				cFrameCpy = cFrame.copy()
				pFrame = self.eyes.fetchPrevious()
				motion,processedframe = self.motionDetected(pFrame,cFrameCpy)

				if(motion):
					#Capture the frame
					log.info("Motion detected")
					self.suspectedIntrustion = self.storeSnapshot(processedframe)
					self.mDetect = True
				else:
					self.mDetect = False

	