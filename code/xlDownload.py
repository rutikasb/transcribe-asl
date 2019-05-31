import openpyxl
import cv2
import urllib.request
import os


book = openpyxl.load_workbook('../data/dai-asllvd-BU_glossing_with_variations_HS_information-extended-urls-RU.xlsx')

sheet = book.active


#EDIT THE FOLLOWING 2 PARAMS TO DECIDE WHICH ROW OF EXCEL FILE TO START AND END AT FOR DOWNLOADING VIDEOS
startRow=3
endRow=35


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
	r = startRow
	while True:
		v = sheet.cell(row=r, column=glossNameCol)
		if(v.value is not None):
			print("\nMain New Gloss: " + v.value)
			videoFolderPath = "../data/videos/" + str(v.value).replace("/",'-')
			if not os.path.exists(videoFolderPath):
				os.makedirs(videoFolderPath)

			print("---------------------")
			i=r+1
			validRows = []
			while True:
				vv = sheet.cell(row=i, column=consultantNameCol)
				if(vv.value == "============"):
					r=i
					break
				else:
					if (vv.value is not None) and (vv.value != '------------'):
						validRows.append(i)
				i += 1
			for i in validRows:
				seq = sheet.cell(row=i, column=sequenceCol)
				scene = sheet.cell(row=i, column=sceneCol)
				frameStart = sheet.cell(row=i, column=frameStartCol)
				frameEnd = sheet.cell(row=i, column=frameEndCol)
				st = str(seq.value) + "/" + "scene" + str(scene.value) + "-" + "camera" + cameraNum + ".mov"
				url = baseURL + st
				print(url)
				filename = str(seq.value) + "-" + "scene" + str(scene.value) + "-" + "camera" + cameraNum + "_start" + str(frameStart.value) + "_" + "end" + str(frameEnd.value) + ".mov"
				filePath = os.path.join(videoFolderPath, filename)
				urllib.request.urlretrieve(url, filePath)
		if r >= endRow:
			break


if __name__ == '__main__':
	downloadVideos()
	# playVideoFile("../data/videos/TWENTY/ASL_2008_05_12a-scene1-camera1_start2400_end2480.mov")