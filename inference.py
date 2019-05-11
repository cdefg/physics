import numpy as np
import cv2 as cv

#显示空间相干性的图像
#原理：大学物理，波动的叠加
#构造一个屏幕，构造光源点，通过光程差计算该点振幅，将振幅叠加得到图形
#可以近似显示双缝干涉、点干涉、圆盘衍射的图像

class LightPoint:
	'''
	对于点光源的描述，如果是线性光源，使用专用的类
	但是多个点光源的近似
	'''
	def __init__(self, k, location:np.ndarray, Amplitude=1, phase=0):
		self.Amplitude = Amplitude
		self.k = k
		self.location = location
		self.phase = phase

	def light_info(self):
		print("[+] Light Point")
		print('    location:{}'.format(self.location))
		print("    k       :{}".format(self.k))
		print("    phase   :{}".format(self.phase))

	def distance_from_point(self, anotherpoint:np.ndarray):
		delta = self.location - anotherpoint
		return np.sqrt(np.dot(delta, delta))

	def intensity(self, distance):
		return self.Amplitude*np.cos(self.phase - self.k*distance)


class Screen:
	'''
	构造一个屏幕，这个屏幕为长方形，可以观察空间内所有的光源造成的图像
	默认位置:在空间平面的yOz平面上
	一般令光源在关于x轴的对称形状上
	'''
	def __init__(self, light_sources:list, height = 600, width = 800):
		self.light_sources = light_sources
		self.height = height
		self.width  = width
		self.center = np.zeros([0, 0, 0])#hardcode, as default
		self.offset = np.array([0, -self.width/2, self.height/2])
		self.canvas = np.zeros((self.height, self.width), dtype = np.float)

	def calculate(self):
		for light in self.light_sources:
			for i in range(self.height):
				for j in range(self.width):
					self.canvas[i, j] += light.intensity(light.distance_from_point(np.array([0, j, -i])+self.offset))
		#归一化处理，便于处理成图像
		#物理基础：对比度/相对照度
		cmax = self.canvas.max()
		cmin = self.canvas.min()
		self.canvas = (self.canvas - cmin)/(cmax - cmin)

	def show(self):
		if (not self.canvas.any()):
			print("[?] have you ever used  .calculate() method?")
		#self.demo = np.zeros((self.height, self.width), dtype = np.int8)
		self.demo = (255*self.canvas).astype(np.int8)
		print(self.canvas)
		cv.imshow('Inference', self.demo)
		cv.waitKey(0)


if __name__ == '__main__':
	l1 = LightPoint(1, np.array([60, -20, 0]))
	l2 = LightPoint(1, np.array([60,  20, 0]))

	ls = [l1, l2]
	
	for l in ls:
		l.light_info()

	s = Screen(ls)
	s.calculate()
	s.show()
