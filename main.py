# Invisible Object- Make Any Object(s) Invisible Simulataneously 
# Author: Savan Visalpara


import cv2
from torchvision import models
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.append("./light")
import argparse
from segment import mmodel

def invisible( video, outv, show):

	cap = cv2.VideoCapture(video)
	# extend the idea presented here: 
	# https://github.com/kaustubh-sadekar/Invisibility_Cloak
	time.sleep(3)
	background=0
	for i in range(30):
		ret,background = cap.read()
		background = cv2.resize(background, (520,350))

	if outv:
		# this can write .avi only
		out = cv2.VideoWriter(outv,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (520,350))

	# list of objects to make invisible
	inv = []
	while True:
		ret, frame = cap.read()
		frame = cv2.resize(frame, (520, 350))
		# cv2.imshow('input', frame)
		
		output = mmodel(frame)
		
		mask = np.zeros((350,520,1))
		for i in inv:
			mask[np.where(output==i)] = 255
		kernel = np.ones((30,30),np.uint8)
		#dilate to include surrounding pixels which might have not be detected
		dilated = cv2.dilate(mask,kernel,iterations = 1)
		# cv2.imshow("dilated", dilated)
		frame[np.where(dilated==255)] = background[np.where(dilated==255)]
	
				
		if outv:
			out.write(frame)
			if show:
				cv2.imshow("frame", frame)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
			


		# if you don't want to remove objects dynamically remove below line and
		# and objects to inv list above
		inp = input("Enter class to make invisible or hit enter to continue:")
		if inp.startswith('c'):
			c = int(inp[1:])
			if c in inv:
				inv.remove(c)
		elif inp == ord('q'):
			break
		elif inp:
			inv.append(int(inp))

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--video", default="IMG_3322_trim.mp4", help="path to input video")
	parser.add_argument("--out_video", default=None,help="path to output video") #if not passes, show on screen
	parser.add_argument("--show", default=True)
	args = parser.parse_args()
	invisible(args.video, args.out_video, args.show)