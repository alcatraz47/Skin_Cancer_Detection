import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Separator():
	def __init__(self):
		self.path = os.getcwd()
		self.target_folder = os.path.join(self.path, 'bkl')
		self.source_folder = os.path.join(self.path, "Ham10000_images_part_1")
		self.training_metadata = pd.read_excel(os.path.join(self.path, 'HAM10000_metadata_excel.xlsx'))

	def get_data(self):
		iterator = 0
		excel_file = self.training_metadata
		for index, row in (excel_file.iterrows()):
			try:
				temp_label = row['dx']
				print(temp_label)
				if temp_label == 'bkl':	
					img_array = cv2.imread((os.path.join(self.source_folder, row['image_id'] + '.jpg')))
					new_array = cv2.resize(img_array, (100, 100))
					cv2.imwrite(os.path.join(self.target_folder, 'bkl' + str(iterator) + '.jpg'), new_array)
					iterator+=1
					#print("try te ashche")
			except Exception as e:
				#print(e)
				pass

if __name__ == '__main__':
	separator = Separator()
	separator.get_data()