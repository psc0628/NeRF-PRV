import numpy
import os
import run
import time

if __name__ == '__main__':
	while True:
		while os.path.isfile('./interact/ready_c++.txt') == False:
			time.sleep(0.1)
		time.sleep(1)
		os.remove('./interact/ready_c++.txt')
		os.system('python ./interact/run_with_c++.py')
		f = open('./interact/ready_py.txt', 'a')
		f.close()

