# Invisible Object- Make Any Object(s) Invisible Simulataneously 
# Author: Savan Visalpara


import cv2
from torchvision import models
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
sys.path.append("./light")
import argparse
from segment import mmodel

def background( video, outv, show, bg, dn):

	cap = cv2.VideoCapture(video)
	bg_img = cv2.imread(bg, cv2.IMREAD_COLOR)
	bg_img = cv2.resize(bg_img, (520, 350))  # resize background image
	cv2.imshow("bg", bg_img)

	if outv:
		# this can write .avi only
		out = cv2.VideoWriter(outv,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (520,350))

	inv = [15]  #objects to keep in foreground
	while True:
		ret, frame = cap.read()
		frame = cv2.resize(frame, (520, 350))
		# cv2.imshow('input', frame)
		
		output = mmodel(frame)
		
		mask = np.zeros((350,520,3))
		for i in inv:
			mask[np.where(output==i)] = 255
		# kernel = np.ones((30,30),np.uint8)
		#dilate to include surrounding pixels which might have not be detected
		# dilated = cv2.dilate(mask,kernel,iterations = 1)
		# cv2.imshow("dilated", dilated)
		dilated = mask
		# cv2.imshow("mask", dilated)
			
		frame[np.where(dilated==0)] = bg_img[np.where(dilated==0)]
				
		if outv:
			out.write(frame)
		if show:
			cv2.imshow("frame", frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		if dn:
			inp = input("Enter new background image or hit enter to continue")
			if os.path.exists(inp):
				bg_img = cv2.imread(inp, cv2.IMREAD_COLOR)
				bg_img = cv2.resize(bg_img, (520, 350))
			else:
				print("Entered image does not exist.")



	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--video", default="./assets/bgdemo3.mp4", help="path to input video")
	parser.add_argument("--out_video", default="bgdemo.avi",help="path to output video") #if not passes, show on screen
	parser.add_argument("--show", default=True)
	parser.add_argument("--bg", default="./assets/default_bg.jpg")
	parser.add_argument("--change_bg_dynamic", default=True, help="allow to change bg while running")
	args = parser.parse_args()
	background(args.video, args.out_video, args.show, args.bg, args.change_bg_dynamic)