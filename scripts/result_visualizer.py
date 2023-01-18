#!/usr/bin/env python3

from cProfile import label
from cmath import sqrt
from tracemalloc import start
from turtle import color

from matplotlib.patches import bbox_artist
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import csv
import sys

class ResultVisualizer:
	def __init__(self,args):
		self.args = args
		self.load_data()
	
	def load_data(self):
		if 2 <= len(self.args):
			print("load file: " + self.args[1])
			self.csv_file = open(self.args[1],"r",encoding="ms932")
			file = csv.reader(self.csv_file,delimiter=",",doublequote=True,lineterminator="\r\n",quotechar='"',skipinitialspace=True)
			


if __name__=='__main__':
	data_viwer = ResultVisualizer(sys.argv)
	