import os
import sys

def main(folder):
	tot = (len(os.listdir(folder))-1)/5
	print(tot)
	for f in os.listdir(folder):
		# print(f)
		if sys.argv[2] != "promo":
			if (f[-4:] == '.txt') & (f[:-4] != 'timeStamp'):
				file = open(os.path.join(folder, f), 'r')
				for line in file:
					print(line.rstrip('\n'))
				print(f)
		else:
			if (f[-4:] == '.txt') & (f[:-4] == 'timeStamp'):
				file = open(os.path.join(folder, f), 'r')
				for line in file:
					if line[:5] == "Promo":
						line = line[6:]
						print(line.rstrip('\n'))

if __name__=='__main__':
	# print('start')
	main(sys.argv[1])
