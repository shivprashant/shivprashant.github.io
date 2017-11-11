from enum import Enum
class ImageTag(Enum):
	Empty = 0,
	Object = 1,
	Family = 2,
	Friend = 3,
	Supplier = 4,
	Intruder = 911

class IntruderDetection:
	def __init__(self,trainingDir,monitoringDir):
		return

	def train(self):
		return True

	def predict(self,img_path):
		return ImageTag.Intruder

	def isFamily(self,img_path):
		return False
