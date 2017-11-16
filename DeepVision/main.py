from securityCameraW10 import SecutiryCameraW10
from watchdog import WatchDog

from facereco import IntruderDetection
#Directory stucture interface for Model training
#trainingDir : Path for training images. Subdir structure as follows
#    bootstrapimages ( images that serve as startup config. Can be empty!!)
#        Empty ( an empty scene with no objects )
# 		 Object ( e.g. Car, Bike, Cycle, Stone)
#		 Family 
#		 Friend
#		 Supplier (day and time based unknown - postman,milkman, gardner)
#		 Intruder (unknow person or suspected unknown)
#	triagedImage ( Human triaged live images for training. Start config empty)
#        Empty
# 		 Object ( e.g. Car, Bike, Cycle, Stone)
#		 Family 
#		 Friend
#		 Supplier (day and time based unknown - postman,milkman, gardner)
#		 Intruder (unknow person or suspected unknown)
# monitoringDir : Path for live images classified by model. Structure as follows
#     <Date> (Date of live image capture)
#	     NoPerson
# 		 Object ( e.g. Car, Bike, Cycle, Stone)
#		 Family 
#		 Friend
#		 Supplier (day and time based unknown - postman,milkman, gardner)
#		 Intruder (unknow person or suspected unknown)
#
# Contents of monitoringDir are not altered. Used as reference for model perf.
# During the triage process the contents are read and may be copied over
# to trainingDir/TriagedImages into correct categories 

rootDir = "C:/Users/shivsood/OneDrive/MyWork/DeepVision"
trainingDir = rootDir + "/training"
monitoringDir = rootDir + "/monitoring"

#intruderLDM - Intruder Learning and Detection Module
intruderLDM = IntruderDetection(trainingDir,monitoringDir)
intruderLDM.train()

eyes = SecutiryCameraW10()
watchdog = WatchDog(eyes,intruderLDM)

watchdog.watch()