import abc

class CameraIf(abc.ABC):
	@abc.abstractmethod
	def __init__(self):
		pass

	def __del__(self):
		pass

	def fetchCurrent(self):
		pass

	def fetchPrevious(self):
		pass

