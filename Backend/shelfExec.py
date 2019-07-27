import sys
import os
import stitch_images as si
# import shelfShareCalc as ss
import testingTensor as tt

def main(sessionId):
	ind = si.main(sessionId)
	# ss.main_fn(sessionId)
	tt.main(sessionId)

if __name__=='__main__':
	# print(sys.argv[1])
	# print(sys.argv[2])
	if (sys.argv[1] == 'result'):
		if (sys.argv[2] == 'table'):
			# data = []
			# for f in os.listdir(sys.argv[3]):
				# filePath = os.path.join(sys.argv[3], f)
			file = open(os.path.join(sys.argv[3], 'timeStamp.txt'), "r")
			for line in file:
				if line[:5] != "Promo":
					print(line.rstrip('\n'))
			# data.append(mylist)
			# print(data)
		else:
			# folders = []
			for f in os.listdir(sys.argv[2]):
				print(f)
	else:
		main(sys.argv[1])
		print("Execution Completed")