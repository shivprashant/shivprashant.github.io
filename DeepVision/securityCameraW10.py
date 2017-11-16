from CameraIf import CameraIf

import cv2
from Logger import log

class SecutiryCameraW10(CameraIf):

	def __init__(self):
		port = 0
		self.camera = cv2.VideoCapture(port)
		self.currentFrame = None
		self.previousFrame = None

	def __del__(self):
		del(self.camera)
		del(self.currentFrame)
		del(self.previousFrame)

	def fetchCurrent(self):
		(grabbed, newFrame) = self.camera.read()

		if(grabbed!=True):
			log.error("Camera not functional!!")
			return False,None

		if self.previousFrame is None:
			self.previousFrame = newFrame
			self.currentFrame  = newFrame
		else:
			del(self.previousFrame)
			self.previousFrame = self.currentFrame
			self.currentFrame = newFrame

		return True,self.currentFrame			

	def fetchPrevious(self):
		return self.previousFrame

	