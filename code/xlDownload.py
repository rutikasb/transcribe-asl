import openpyxl
import cv2
import urllib.request
import os


book = openpyxl.load_workbook('../data/dai-asllvd-BU_glossing_with_variations_HS_information-extended-urls-RU.xlsx')

sheet = book.active

startRow=3
endRow=5
glossNameCol = 2
consultantNameCol = 3
sequenceCol = 13
sceneCol = 14
frameStartCol = 15
frameEndCol = 16
baseURL = 'http://csr.bu.edu/ftp/asl/asllvd/asl-data2/quicktime/'
cameraNum = "1"


def playVideoFile(filename):
	cap = cv2.VideoCapture(filename)
	while(cap.isOpened()):
	    ret, frame = cap.read()
	    cv2.imshow('frame',frame)
	    if cv2.waitKey(1) & 0xFF == ord('q'):
	        break

	cap.release()
	cv2.destroyAllWindows()


def downloadVideos():
	for r in range(startRow, endRow):
		v = sheet.cell(row=r, column=glossNameCol)
		if(v.value is not None):
			print("\nMain New Gloss: " + v.value)
			videoFolderPath = "../data/videos/" + str(v.value)
			if not os.path.exists(videoFolderPath):
				os.makedirs(videoFolderPath)

			print("---------------------")
			i=r+1
			consultantStartRow, consultantEndRow = i,i
			while True:
				vv = sheet.cell(row=i+1, column=consultantNameCol)
				if(vv.value == "============"):
					consultantEndRow = i
					break
				i += 1
			# print("Consultant start row = {0}, end row = {1}".format(consultantStartRow, consultantEndRow))
			for i in range(consultantStartRow, consultantEndRow+1):
				seq = sheet.cell(row=i, column=sequenceCol)
				scene = sheet.cell(row=i, column=sceneCol)
				frameStart = sheet.cell(row=i, column=frameStartCol)
				frameEnd = sheet.cell(row=i, column=frameEndCol)
				st = str(seq.value) + "/" + "scene" + str(scene.value) + "-" + "camera" + cameraNum + ".mov"
				url = baseURL + st
				print(url)
				filename = str(seq.value) + "-" + "scene" + str(scene.value) + "-" + "camera" + cameraNum + "_start" + str(frameStart.value) + "_" + "end" + str(frameEnd.value) + ".mov"
				filePath = os.path.join(videoFolderPath, filename)
				print(filePath)
				urllib.request.urlretrieve(url, filePath)


if __name__ == '__main__':
	# downloadVideos()
	playVideoFile("/home/chandangope/ucb-mids/w210/transcribe-asl/data/videos/TWENTY/ASL_2008_05_12a-scene1-camera1_start2400_end2480.mov")