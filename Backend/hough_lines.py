import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import matplotlib.path as mplPath
import os
from scipy import ndimage
from PIL import Image

def Slope(x0, y0, x1,y1):
	return ((float)(y1-y0)/(x1-x0))

def fullLine(img, a, b):
	slope=0.0
	#print(b[0]-a[0])
	if((b[0]-a[0])!=0):
		slope = Slope(a[0], a[1], b[0], b[1])
	else:
		print("infinity slope") 
		return None	
	p=[0,0] 
	q=[img.shape[1],img.shape[0]]
	
	p[1] = int(-(a[0] - p[0]) * slope + a[1])
	q[1] = int(-(b[0] - q[0]) * slope + b[1])

	#cv2.line(img,tuple(p),tuple(q),(0,255,0),1)
	return(p,q)

def point_inside_polygon(x,y,poly):

    n = len(poly)
    inside =False

    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xinters = float(float(y-p1y)*float(p2x-p1x)/float(p2y-p1y)+p1x)
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside

def dist(a,b):
	return(math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2))

count=0
folder_name='/home/hemanthkumaran/scrap_images/'

for filename in os.listdir(folder_name):
	file_str=str(folder_name+filename)	
	img = cv2.imread(file_str,1)
	#print(img.shape)
	if(img.shape[0]<512 and img.shape[1]<512):
		img = cv2.resize(img,(512,512),interpolation = cv2.INTER_CUBIC)
	#img = cv2.resize(img,(2*img.shape[0],2*img.shape[1]),interpolation = cv2.INTER_CUBIC)	
	gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
	edges = cv2.Canny(gray,50,450,apertureSize = 3)

	# plt.subplot(121),plt.imshow(img,cmap = 'gray')
	# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
	# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
	# plt.show()

	minLineLength = 10
	maxLineGap = 20
	lines = cv2.HoughLinesP(edges,1,(np.pi/180.0),10,minLineLength,maxLineGap)
	#fullLine(img,(0,2),(2,4),'r')

	# print(lines.size)
	# print(lines)

	lst=[]
	lst.append(((0,0),(img.shape[1],0)))
	pos_count=0
	neg_count=0
	zero_count=0
	slope_type=''
	if(lines==None or len(lines)==0):
		continue
		print('here',file_str)	

	for line in lines:
		for x1,y1,x2,y2 in line:
			#cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
			slope=Slope(x1,y1,x2,y2)
			if((x2-x1)==0 or math.fabs(slope)>1):
				continue
			#print(slope)
			if((x2-x1)!=0 and slope==0.0):
				zero_count=zero_count+1
			if((x2-x1)!=0 and slope>0.0):
				pos_count=pos_count+1
			else:
				if((x2-x1)!=0 and slope<0.0):
					neg_count=neg_count+1	

	print(pos_count,neg_count,zero_count)			
	if(math.fabs(pos_count-neg_count)<=2 or (zero_count>pos_count+10 and zero_count>neg_count+10)):
		print(filename,'neutral')
		neutral=True
		slope_type='neutral'
	elif(pos_count>(neg_count)):
		positive=True
		slope_type='positive'
		print(filename,'positive')
	else:
		print(filename,'negative')
		slope_type='negative'
		positive=False
	
	for line in lines:
		for x1,y1,x2,y2 in line:
			#print(slope)
			#fullLine(img,(x1,y1),(x2,y2))
			if(slope_type=='neutral'):
				slope=Slope(x1,y1,x2,y2)
				#print(slope)
				if(math.fabs(slope)<0.1):
						#print(fullLine(img,(x1,y1),(x2,y2)))
						# cv2.line(img,(x1,y1),(x2,y2),(0,0,255),4)						
						# cv2.putText(img, str(slope), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255))
						# (p,q)=fullLine(img,(x1,y1),(x2,y2))
						# print("ggg",slope,math.fabs(p[1]-q[1]))
						# if(fullLine(img,(x1,y1),(x2,y2))!=None and (math.fabs(p[1]-q[1])<240)):
						lst.append(fullLine(img,(x1,y1),(x2,y2)))

			if(slope_type=='positive'):
				slope=Slope(x1,y1,x2,y2)
				if(slope<0.6 and slope>=0):
						#print(fullLine(img,(x1,y1),(x2,y2)))
						# cv2.line(img,(x1,y1),(x2,y2),(0,0,255),4)						
						# cv2.putText(img, str(slope), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255))
						# (p,q)=fullLine(img,(x1,y1),(x2,y2))
						# print("ggg",slope,math.fabs(p[1]-q[1]))
						# if(fullLine(img,(x1,y1),(x2,y2))!=None and (math.fabs(p[1]-q[1])<240)):
						lst.append(fullLine(img,(x1,y1),(x2,y2)))
			else:
				slope=Slope(x1,y1,x2,y2)
				if(slope>-0.6 and slope<=0):
						#print(fullLine(img,(x1,y1),(x2,y2)))
						# cv2.line(img,(x1,y1),(x2,y2),(0,0,255),4)						
						# cv2.putText(img, str(slope), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255))
						# (p,q)=fullLine(img,(x1,y1),(x2,y2))
						# print("ggg",slope,math.fabs(p[1]-q[1]))
						# if(fullLine(img,(x1,y1),(x2,y2))!=None and (math.fabs(p[1]-q[1])<240)):
						lst.append(fullLine(img,(x1,y1),(x2,y2)))
				# slope=Slope(x1,y1,x2,y2)
				# if(math.fabs(slope)<0.6):
				# 		#print(fullLine(img,(x1,y1),(x2,y2)))
				# 		cv2.line(img,(x1,y1),(x2,y2),(0,0,255),4)
						
				# 		cv2.putText(img, str(slope), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255))
				# 		(p,q)=fullLine(img,(x1,y1),(x2,y2))
				# 		print("ggg",slope,math.fabs(p[1]-q[1]))
				# 		if(fullLine(img,(x1,y1),(x2,y2))!=None and (math.fabs(p[1]-q[1])<240)):
				# 			lst.append(fullLine(img,(x1,y1),(x2,y2)))					

	#cv2.imshow('houghlines5.jpg',img)
	# #print(folder_name+'slope/'+filename)
	#cv2.imwrite(folder_name+'slope/'+slope_type+filename,img)
	#cv2.waitKey(0)


	# #cv2.namedWindow("img", cv2.WINDOW_NORMAL)
	lst_of_lst=[]
	lst_of_lst_flag=[]
	temp_lst=[]
	first_cord=0
	second_cord=0
	sum_first_cord=0
	sum_second_cord=0
	lines_lst=[]
	if(slope_type!='positive'):		
		def getKey1(item):
			return item[0][1]
			
		lst = sorted(lst,key=getKey1)

		prev_cord=lst[0][0][1]
		temp_lst.append([(0,lst[0][0][1]),(img.shape[0],lst[0][1][1])])
	
		for n in range(len(lst)):
			if(lst[n][0][1]-prev_cord<60):
				temp_lst.append([(lst[n][0][0],lst[n][0][1]),(lst[n][1][0],lst[n][1][1])])
			else:
				first_cord=temp_lst[(len(temp_lst)-1)][0]
				second_cord=temp_lst[(len(temp_lst)-1)][1]
				cv2.line(img,first_cord,second_cord,(0,255,0),2)
				if(len(lst_of_lst)>=1):
					#print("len > 1",len(lst_of_lst))
					prev_first_cord,prev_second_cord=lst_of_lst[len(lst_of_lst)-1]
					#print(Slope(first_cord[0],first_cord[1],second_cord[0],second_cord[1]))
					
					# for i in range(len(lst_of_lst)):
					# 	sum_first_cord[0]+=lst_of_lst[i][0][0]
					# 	sum_first_cord[1]+=lst_of_lst[i][0][1]
					# 	sum_second_cord[0]+=lst_of_lst[i][1][0]
					# 	sum_second_cord[1]+=lst_of_lst[i][1][1]
					#print("appending to flag",len(lst_of_lst_flag))

					if(dist(prev_first_cord,first_cord)>300):
						#print(dist(prev_first_cord,first_cord))
						#print("in inner")
						# cv2.imshow('/home/hemanthkumaran/planogram/test/found/'+str(i)+'_'+str(count)+'.jpeg',img)
						# cv2.waitKey(0)
						# cv2.destroyAllWindows()
						count=count+1
						lst_of_lst_flag.append(True)
						#print(lst_of_lst_flag)
					else:
						lst_of_lst_flag.append(False)	

				lst_of_lst.append([first_cord,second_cord])
				lines_lst.append([first_cord,second_cord])	
				#print("appending to lst",len(lst_of_lst))
				#print('\n')

				temp_lst=[]
				temp_lst.append([(lst[n][0][0],lst[n][0][1]),(lst[n][1][0],lst[n][1][1])])
			#print((lst_of_lst_flag))
			#print(lst_of_lst)
			prev_cord=lst[n][0][1]		
		
		#print(len(lst_of_lst_flag))
		#print(len(lst_of_lst))
			
		if(len(temp_lst)!=0):
				first_cord=temp_lst[(len(temp_lst)-1)][0]
				second_cord=temp_lst[(len(temp_lst)-1)][1]	
				cv2.line(img,first_cord,second_cord,(0,255,0),2)
				#lst_of_lst_flag.append(True)
				#print(Slope(first_cord[0],first_cord[1],second_cord[0],second_cord[1]))
				if(len(lst_of_lst)>=1):
					prev_first_cord,prev_second_cord=lst_of_lst[len(lst_of_lst)-1]
					
					if(dist(prev_first_cord,first_cord)>300):
						#print(dist(prev_first_cord,first_cord))
						#print("in inner 1")
						# cv2.imshow('/home/hemanthkumaran/planogram/test/found/'+str(i)+'_'+str(count)+'.jpeg',img)
						# cv2.waitKey(0)
						# cv2.destroyAllWindows()
						count=count+1
						lst_of_lst_flag.append(True)	
						#print(lst_of_lst_flag)
					else:
						lst_of_lst_flag.append(False)	

				lst_of_lst.append([first_cord,second_cord])	
				lines_lst.append([first_cord,second_cord])		
				#print(first_cord,second_cord)
				#print(second_cord)
		if(len(lst_of_lst)>=1):
					prev_first_cord,prev_second_cord=lst_of_lst[len(lst_of_lst)-1]
					#print(dist(prev_first_cord,(0,img.shape[0])))
					first_cord=(0,img.shape[0])
					second_cord=(img.shape[1],img.shape[0])
					#print(Slope(first_cord[0],first_cord[1],second_cord[0],second_cord[1]))
					if(dist(prev_first_cord,first_cord)>300):
						#print("in outer")
						# cv2.imshow('/home/hemanthkumaran/planogram/test/found/'+str(i)+'_'+str(count)+'.jpeg',img)
						# cv2.waitKey(0)
						# cv2.destroyAllWindows()
						count=count+1
						lst_of_lst_flag.append(True)				
					else:
						lst_of_lst_flag.append(False)	
		lst_of_lst.append([(0,img.shape[0]),(img.shape[1],img.shape[0])])	
		cv2.line(img,(0,img.shape[0]),(img.shape[1],img.shape[0]),(0,255,0),2)		
	else:
		def getKey1(item):
			return item[1][1]
			
		lst = sorted(lst,key=getKey1)
		#for item in lst:
		#print(item)		
		prev_cord=lst[0][1][1]
		temp_lst.append([(img.shape[0],lst[0][1][1]),(0,lst[0][0][1])])
		for n in range(len(lst)):
			#print(prev_cord,lst[n][1][1])
			if(lst[n][1][1]-prev_cord<60):
				temp_lst.append([(lst[n][1][0],lst[n][1][1]),(lst[n][0][0],lst[n][0][1])])
				#first_cord+=lst[n][1][1]
				#second_cord+=lst[n][0][1]
			else:
				#print(temp_lst)
				#lst_of_lst.append(temp_lst)
				first_cord=temp_lst[(len(temp_lst)-1)][0]
				second_cord=temp_lst[(len(temp_lst)-1)][1]
				cv2.line(img,first_cord,second_cord,(0,255,0),2)
				#print(Slope(first_cord[0],first_cord[1],second_cord[0],second_cord[1]))
				if(len(lst_of_lst)>=1):
					#print("len > 1",len(lst_of_lst))
					prev_first_cord,prev_second_cord=lst_of_lst[len(lst_of_lst)-1]
					
					#print("appending to flag",len(lst_of_lst_flag))
					if(dist(prev_first_cord,first_cord)>300):
						# cv2.imshow('/home/hemanthkumaran/planogram/test/found/'+str(i)+'_'+str(count)+'.jpeg',img)
						# cv2.waitKey(0)
						# cv2.destroyAllWindows()
						count=count+1
						lst_of_lst_flag.append(True)
					else:
						lst_of_lst_flag.append(False)	
								
				lst_of_lst.append([first_cord,second_cord])			
				#print(first_cord,second_cord)
				#print(second_cord)
				# first_cord=0
				# second_cord=0
				# first_cord+=lst[n][1][1]
				# second_cord+=lst[n][0][1]
				temp_lst=[]
				temp_lst.append([(lst[n][1][0],lst[n][1][1]),(lst[n][0][0],lst[n][0][1])])

			prev_cord=lst[n][1][1]					
		if(len(temp_lst)!=0):
				first_cord=temp_lst[(len(temp_lst)-1)][0]
				second_cord=temp_lst[(len(temp_lst)-1)][1]
				cv2.line(img,first_cord,second_cord,(0,255,0),2)
				#print(Slope(first_cord[0],first_cord[1],second_cord[0],second_cord[1]))
				if(len(lst_of_lst)>=1):
					#print("len > 1",len(lst_of_lst))
					prev_first_cord,prev_second_cord=lst_of_lst[len(lst_of_lst)-1]
					
					#print("appending to flag",len(lst_of_lst_flag))
					if(dist(prev_first_cord,first_cord)>300):
						# cv2.imshow('/home/hemanthkumaran/planogram/test/found/'+str(i)+'_'+str(count)+'.jpeg',img)
						# cv2.waitKey(0)
						# cv2.destroyAllWindows()
						count=count+1
						lst_of_lst_flag.append(True)
					else:
						lst_of_lst_flag.append(False)	
				lst_of_lst.append([first_cord,second_cord])		
				#print(first_cord,second_cord)
				#print(second_cord)
		
		if(len(lst_of_lst)>=1):
					#print("len > 1",len(lst_of_lst))
					prev_first_cord,prev_second_cord=lst_of_lst[len(lst_of_lst)-1]
					first_cord=(img.shape[1],img.shape[0])
					second_cord=(0,img.shape[0])
					#print(Slope(first_cord[0],first_cord[1],second_cord[0],second_cord[1]))
					#print("appending to flag",len(lst_of_lst_flag))
					#print((prev_first_cord,first_cord))
					#print(dist(prev_first_cord,first_cord))
					if(dist(prev_first_cord,first_cord)>300):
						# cv2.imshow('/home/hemanthkumaran/planogram/test/found/'+str(i)+'_'+str(count)+'.jpeg',img)
						# cv2.waitKey(0)
						# cv2.destroyAllWindows()
						#print("outer")
						count=count+1
						lst_of_lst_flag.append(True)
					else:
						lst_of_lst_flag.append(False)	

		lst_of_lst.append([(img.shape[1],img.shape[0]),(0,img.shape[0])])		
		cv2.line(img,(img.shape[1],img.shape[0]),(0,img.shape[0]),(0,255,0),2)			
	
	#for line in lst_of_lst:
	#	print(Slope(line[0][0],line[0][1],line[1][0],line[1][1]))
	cv2.imshow('img',img)
	#cv2.imwrite(folder_name+'slope/'+filename,img)
	cv2.waitKey(0)
	#print(len(lst_of_lst_flag),len(lst_of_lst))
	#for m in range(len(lst_of_lst_flag)):
	#   	print(lst_of_lst[m])
	#   	print(lst_of_lst_flag[m])

# 	path_lst=[]
# 	path_lst1=[]
# 	slope_lst=[(0.0,0.0)]
# 	#print(lst_of_lst_flag)	
# 	for m in range(0,len(lst_of_lst)-1):
# 			arr=np.array([[lst_of_lst[m][0][1],lst_of_lst[m][0][0]], [lst_of_lst[m][1][1],lst_of_lst[m][1][0]],
# 						 [lst_of_lst[m+1][1][1],lst_of_lst[m+1][1][0]],[lst_of_lst[m+1][0][1],lst_of_lst[m+1][0][0]], 
# 						 [lst_of_lst[m][0][1],lst_of_lst[m][0][0]]])
# 	# 		h=(int)(math.fabs(lst_of_lst[m][1][1]- lst_of_lst[m+1][1][1]))
# 	# 		w=(int)(math.fabs(lst_of_lst[m][0][0] - lst_of_lst[m][1][0]))
# 	# 		print(w,h)
# 	# 		print((lst_of_lst[m][0][0],lst_of_lst[m][0][1]))
# 	# 		print((lst_of_lst[m][0][0]+w,lst_of_lst[m][0][1]+h))
# 	# 		cv2.rectangle(img,(lst_of_lst[m][0][0],lst_of_lst[m][0][1]),(lst_of_lst[m][0][0]+w,lst_of_lst[m][0][1]+h)
# 	# 			,(0,255,0),2)
# 	# 		cv2.imshow('img',img)
# 	# #cv2.imwrite(folder_name+'slope/'+filename,img)
# 	# 		cv2.waitKey(0)
			
# 	# 		#rect = cv2.minAreaRect(arr)
# 	# 		#box = cv2.boxPoints(rect)
# 	# 		#print(arr)
# 	# 		# print(rect)
# 	# 		# # mask = np.zeros((img.shape[0]),np.uint8)
# 	# 		# box = np.int0(box)
# 	# 		# cv2.drawContours(img,[box],0,(0,0,255),2)	
# 	# 		# cv2.imshow('img',img)
# 	# 		# cv2.waitKey(0)

# 			if(m!=0):
# 				print(lst_of_lst[m][0][0],lst_of_lst[m][0][1],lst_of_lst[m][1][0],lst_of_lst[m][1][1])
# 				print(lst_of_lst[m+1][0][0],lst_of_lst[m+1][0][1],lst_of_lst[m+1][1][0],lst_of_lst[m+1][1][1])				
# 				slope1=(Slope(lst_of_lst[m][0][0],lst_of_lst[m][0][1],lst_of_lst[m][1][0],lst_of_lst[m][1][1]))
# 				slope2=(Slope(lst_of_lst[m+1][0][0],lst_of_lst[m+1][0][1],lst_of_lst[m+1][1][0],lst_of_lst[m+1][1][1]))
# 				slope_lst.append([slope1,slope2])
# 				print([slope1,slope2])
# 			#print(arr)
# 			#if(lst_of_lst_flag[m]):
# 			#	path_lst1.append(mplPath.Path(arr))
# 				#print(arr)
# 				#print("exclu")
# 			#else:		
# 			path_lst.append(mplPath.Path(arr))		
# 				#print("inclu")

		
	
# 	mask = np.zeros((img.shape[0],img.shape[1]))
# 	img1=np.zeros((len(path_lst)+1,img.shape[0],img.shape[1],3),np.uint8)
# 	# # img2=np.zeros((len(path_lst1)+1,img.shape[0],img.shape[1],3),np.uint8)

# 	# # # count=0	
# 	# # for m in range(0,len(slope_lst)):
# 	# # 	#print(m,(slope_lst[m]))
# 	# # 	print()
# 	for m in range(img.shape[0]):
# 			for n in range(img.shape[1]):
# 				for k in range(len(path_lst)):
# 					#if(point_inside_polygon(i,j,path_lst[k])):
# 					if(path_lst[k].contains_point((m,n))):
# 						mask[m][n]=k+1
# 						img1[mask[m][n]][m][n][0]=img[m][n][0]
# 						img1[mask[m][n]][m][n][1]=img[m][n][1]
# 						img1[mask[m][n]][m][n][2]=img[m][n][2]

# 	# # if(len(path_lst1)>0):
# 	# # 	mask = np.zeros((img.shape[0],img.shape[1]))
# 	# # 	for m in range(img.shape[0]):
# 	# # 			for n in range(img.shape[1]):
# 	# # 				for k in range(len(path_lst1)):
# 	# # 					#if(point_inside_polygon(i,j,path_lst[k])):
# 	# # 					if(path_lst1[k].contains_point((m,n))):
# 	# # 						mask[m][n]=k+1
# 	# # 						img2[mask[m][n]][m][n][0]=img[m][n][0]
# 	# # 						img2[mask[m][n]][m][n][1]=img[m][n][1]
# 	# # 						img2[mask[m][n]][m][n][2]=img[m][n][2]	
# 	# # print("done mapping points")				
	

# 	for m in range(1,len(path_lst)+1):
# 	# # 			#	print(img1[i:i+1,:,:,:])
# 					angle1=math.degrees(math.atan(slope_lst[m-1][0]))
# 					angle2=math.degrees(math.atan(slope_lst[m-1][1]))
# 	# 				# angle=0.0
# 					if(math.fabs(angle1)<math.fabs(angle2)):
# 						angle=-angle1
# 					else:
# 						angle=-angle2	
# 					print(img1[m:m+1,].shape)	
# 					print(img.shape)
# 					img3=img1[m:m+1,].reshape(img1.shape[1],img1.shape[2],img1.shape[3])
# 					img3 = ndimage.rotate(img3,-angle)
# 					print(img3.shape)
# 					#rotated = rotated.reshape((rotated.shape[1],rotated.shape[2],rotated.shape[3]))
# 					Image.fromarray(img3,'RGB').show()
# 					#cv2.imshow('img',img1[m:m+1,])
# 					#cv2.waitKey(0)

# 	#for m in range(1,len(path_lst1)+1):
# 	# 			#	print(img1[i:i+1,:,:,:])
# 	# 				cv2.imwrite('/home/hemanthkumaran/planogram/test/found1/img_'+str(i)+'_'+str(m)+'.jpeg',img2[m:m+1,].reshape(img.shape)) 				
# 					#cv2.waitKey(0)
# 						#cv2.imwrite('/home/hemanthkumaran/planogram/test/img.jpeg',img)				
# 				# if(found==False):
# 				# 	mask[i][j]=-1
# 				#else:
# 				#	img1[mask[i][j]][i][j][0]=img[i][j][0]
# 				#	img1[mask[i][j]][i][j][1]=img[i][j][1]
# 				#	img1[mask[i][j]][i][j][2]=img[i][j][2]					
# 	#for i in range(len(lst_of_lst)):
# #		print(lst_of_lst[i])

# 	#cv2.imshow('/home/hemanthkumaran/planogram/test/houghlines5_1.jpg',img)
# 	#cv2.imwrite('/home/hemanthkumaran/planogram/test/2/'+str(i)+'_averaged.jpg',img)
# 	#cv2.waitKey(0)	
# 		#print(lst_of_lst1[i])	
# 	# path_lst=[]

# # for i in range(0,len(lst)-1):
# # 	#print(lst[i][0][0],lst[i+1][1][0],lst[i][0][1],lst[i+1][1][1])
# # 	arr=np.array([[lst[i][0][1],lst[i][0][0]], [lst[i][1][1],lst[i][1][0]],[lst[i+1][1][1],lst[i+1][1][0]], [lst[i+1][0][1],lst[i+1][0][0]],[lst[i][0][1],lst[i][0][0]]])
# # 	#print(arr)
# # 	#hull = cv2.convexHull(arr)
# # 	#print(hull)
# # 	bbPath = mplPath.Path(arr)
# # 	path_lst.append(bbPath)
# 	#print("here")
# 	# bbPath.contains_point((200, 100))
# 	# rect = cv2.minAreaRect(arr)
# 	# box = cv2.boxPoints(rect)
# 	# #print(rect)
# 	# # mask = np.zeros((img.shape[0]),np.uint8)
# 	# box = np.int0(box)
# 	# cv2.drawContours(img,[box],0,(0,0,255),2)	
# 	# #print(box)
# 	# y1 = min(box[2][0],img.shape[0])
# 	# y2 = min(box[0][0],img.shape[0])
# 	# x1 = min(box[2][1],img.shape[1])
# 	# x2 = min(box[0][1],img.shape[1])
# 	# print(x1)
# 	# print(x2)
# 	# print(y1)
# 	# print(y2)
# 	# print('\n')
# 	# #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)	
# 	# cv2.imwrite('/home/hemanthkumaran/planogram/test/rect/'+str(i)+'.jpeg',img[x1:x2,y1:y2])
# 	# cv2.waitKey(0)
# # #print(len(path_lst))

# # mask = np.zeros((img.shape[0],img.shape[1]))
# # img1=np.zeros((len(path_lst)+1,img.shape[0],img.shape[1],3),np.uint8)
# # count=0
# # for i in range(img.shape[0]):
# # 	for j in range(img.shape[1]):
# # 		found=False
# # 		for k in range(len(path_lst)):
# # 			#if(point_inside_polygon(i,j,path_lst[k])):
# # 			if(path_lst[k].contains_point((i,j))):
# # 				found=True
# # 				mask[i][j]=k+1
# # 				img1[mask[i][j]][i][j][0]=img[i][j][0]
# # 				img1[mask[i][j]][i][j][1]=img[i][j][1]
# # 				img1[mask[i][j]][i][j][2]=img[i][j][2]

# # 		# if(found==False):
# # 		# 	mask[i][j]=-1
# # 		#else:
# # 		#	img1[mask[i][j]][i][j][0]=img[i][j][0]
# # 		#	img1[mask[i][j]][i][j][1]=img[i][j][1]
# # 		#	img1[mask[i][j]][i][j][2]=img[i][j][2]				
# # # #
# # #mask[mask !=0] = 255


# # # for i in range(mask.shape[0]):
# # # 	for j in range(mask.shape[1]):
# # # 		if(mask[i][j]!=-1):
# # # 			#print(img[i][j][0],img[i][j][1],img[i][j][2])
# # # 			img1[mask[i][j]][i][j][0]=img[i][j][0]
# # # 			img1[mask[i][j]][i][j][1]=img[i][j][1]
# # # 			img1[mask[i][j]][i][j][2]=img[i][j][2]

# # #print(img1[0:1,].reshape(img.shape).shape)

# # for i in range(1,len(path_lst)):
# # #	print(img1[i:i+1,:,:,:])
# # 	cv2.imwrite('/home/aiml/hemanth/selected/training/rect/'+str(i)+'.jpeg',img1[i:i+1,].reshape(img.shape))
# 	#cv2.waitKey(0)
# 	#cv2.imwrite('/home/hemanthkumaran/planogram/test/img.jpeg',img)

# # #cv2.imwrite('/home/hemanthkumaran/planogram/test/img.jpeg',img)
# # # print(count)
# # #np.savetxt('/home/hemanthkumaran/planogram/test/block.csv',mask,delimiter='\t')

# # # #cv2.waitKey(0)
# # #cv2.imshow('houghlines5.jpg',img)
# # #cv2.imwrite('/home/hemanthkumaran/planogram/test/houghlines5.jpg',img)
# # #cv2.waitKey(0)