from picamera import PiCamera
from picamera.array import PiRGBArray
import cv2

from Logger import log

class SecutiryCameraPi:

	def __init__(self):
		port = 0
		self.camera = PiCamera()
		self.camera.resolution = (640,480)
		self.rawCapture  = PiRGBArray(self.camera,size=(640,480))
		self.currentFrame = None
		self.previousFrame = None
		self.fetchCount = 0

	def __del__(self):
                del(self.camera)
                del(self.rawCapture)
                del(self.currentFrame)
                del(self.previousFrame)

	def fetchCurrent(self):                
                self.fetchCount = self.fetchCount + 1
                if(self.fetchCount == 1):
                    log.info("counter 1")
                    
                if(self.fetchCount == 2):
                    log.info("counter 2")
                
                    
                try:
                    self.camera.capture(self.rawCapture,format="bgr")
                except ValueError as ex:
                    log.error("Exception in camera capture!!")
                    return False,None                
                newFrame = self.rawCapture.array
                self.rawCapture  = PiRGBArray(self.camera,size=(640,480))
                #cv2.imshow("Image",newFrame)
                #exit(0)
                if self.previousFrame is None:
                    self.previousFrame = newFrame
                    self.currentFrame  = newFrame
                else:
                    del(self.previousFrame)
                    
                    #self.rawCapture.truncate(0)
                    #self.rawCapture.seek(0)
                                        
                    self.previousFrame = self.currentFrame
                    self.currentFrame = newFrame
		
                return True,self.currentFrame

	def fetchPrevious(self):
		return self.previousFrame

	
