import sys
import cv2
import numpy as np
import os
import errno

fileLog = open('myapp.txt','a')

# Use the keypoints to stitch the images
def get_stitched_image(img1, img2, M):

  # Get width and height of input images  
  w1,h1 = img1.shape[:2]
  w2,h2 = img2.shape[:2]

  # Get the canvas dimesions
  img1_dims = np.float32([ [0,0], [0,w1], [h1, w1], [h1,0] ]).reshape(-1,1,2)
  img2_dims_temp = np.float32([ [0,0], [0,w2], [h2, w2], [h2,0] ]).reshape(-1,1,2)


  # Get relative perspective of second image
  img2_dims = cv2.perspectiveTransform(img2_dims_temp, M)
  #print(img1_dims,img2_dims)
  # Resulting dimensions
  result_dims = np.concatenate( (img1_dims, img2_dims), axis = 0)

  # Getting images together
  # Calculate dimensions of match points
  [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
  [x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)
  
  # Create output array after affine transformation 
  transform_dist = [-x_min,-y_min]
  transform_array = np.array([[1, 0, transform_dist[0]], 
                [0, 1, transform_dist[1]], 
                [0,0,1]]) 
  #print(x_max-x_min, y_max-y_min)
  # Warp images to get the resulting image
  result_img = cv2.warpPerspective(img2, transform_array.dot(M), 
                  (x_max-x_min, y_max-y_min))
  result_img[transform_dist[1]:w1+transform_dist[1], 
        transform_dist[0]:h1+transform_dist[0]] = img1

  # Return the result
  return result_img

# Find SIFT and return Homography Matrix
def get_sift_homography(img1, img2):

  # Initialize SIFT 
  sift = cv2.xfeatures2d.SIFT_create()

  # Extract keypoints and descriptors
  k1, d1 = sift.detectAndCompute(img1, None)
  k2, d2 = sift.detectAndCompute(img2, None)

  # Bruteforce matcher on the descriptors
  bf = cv2.BFMatcher()
  matches = bf.knnMatch(d1,d2, k=2)

  # Make sure that the matches are good
  verify_ratio = 0.8 # Source: stackoverflow
  verified_matches = []
  for m1,m2 in matches:
    # Add to array only if it's a good match
    if m1.distance < 0.8 * m2.distance:
      verified_matches.append(m1)

  # Mimnum number of matches
  #min_matches = 1000
  fileLog.write("VERIFIED MATCHES: "+str(len(verified_matches)))
  #if len(verified_matches) > min_matches:
    
  # Array to store matching points
  img1_pts = []
  img2_pts = []

  # Add matching points to array
  for match in verified_matches:
    img1_pts.append(k1[match.queryIdx].pt)
    img2_pts.append(k2[match.trainIdx].pt)
  img1_pts = np.float32(img1_pts).reshape(-1,1,2)
  img2_pts = np.float32(img2_pts).reshape(-1,1,2)
  
  # Compute homography matrix
  M, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)
  return (M,len(verified_matches))
  #else:
  # print('Error: Not enough matches')
  # exit()

# Equalize Histogram of Color Images
def equalize_histogram_color(img):
  img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
  img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
  img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
  return img

def stitch_two(img1,img2):
  img1 = equalize_histogram_color(img1)
  img2 = equalize_histogram_color(img2)
  M,len_verified_matches =  get_sift_homography(img1, img2)
  stitch_res = get_stitched_image(img2, img1, M)
  return (stitch_res,len_verified_matches)
# Main function definition

def one_stitch_iter(filepath, img1, img2_idx, stitch_out = [], stitchIndex = [], size_thresh = 0.8):
  #stitch_out=[]
  len_verified_matches_lst = []
  temp_stitch_lst = []
  min_matches = 1000
  # print("inone img2_idx", img2_idx)
  
  for i in img2_idx:
    img2 = cv2.imread(filepath[i])
    temp_stitch, len_verified_matches = stitch_two(img1, img2)
    #print(temp_stitch.shape,img1.shape,img2.shape)
    if(temp_stitch.shape[1] > size_thresh*(img1.shape[1] + img2.shape[1]) and len_verified_matches > min_matches):
      len_verified_matches_lst.append(len_verified_matches)
    else:
      len_verified_matches_lst.append(-1)
    temp_stitch_lst.append(temp_stitch)

  max_len_verified_matches, max_len_verified_matches_idx = np.max(len_verified_matches_lst), np.argmax(len_verified_matches_lst)
  
  if(max_len_verified_matches > -1):
    add_stitch_res = True
    img1 = temp_stitch_lst[max_len_verified_matches_idx]
    stitchIndex.append(img2_idx[max_len_verified_matches_idx])
    img2_idx.remove(img2_idx[max_len_verified_matches_idx])
    stitch_out.append(temp_stitch_lst[max_len_verified_matches_idx])
  
    if(len(img2_idx) != 0):
      stitch_out, img2_idx, stitchIndex = one_stitch_iter(filepath, img1, img2_idx, stitch_out, stitchIndex)

  return (stitch_out, img2_idx, stitchIndex)

def main(filepath):
  path = os.path.join('/data1/naquib.alam/images', filepath)

  #Get number of input images
  filelist = []

  for f in os.listdir(path):
    filelist.append(os.path.join(path, f))
  n_images = len(filelist)
  filelist = sorted(filelist)
  # print(filelist)
  # print(filelist)

  # img1 = cv2.imread(filepath[0])
  res_images = []
  res_img_names = []
  stitch_possible = True
  cnt = 0
  # Get input set of images
  img2_idx = list(range(0, n_images))
  stitch_res = []
  # while True:
  stitch_indices = []
  while len(img2_idx) > 1:
    # print("ID:", img2_idx)
    
    img1 = cv2.imread(filelist[img2_idx[0]])
    temp = [img2_idx[0]]
    img2_idx.pop(0)
    # print("IDX:", img2_idx)
    stitch_out, img2_idx, stitch_index_temp = one_stitch_iter(filelist, img1, img2_idx, [], temp)
    # print("stitch_index_temp: ",stitch_index_temp)
    if(len(stitch_out) > 0):
      stitch_res.append(stitch_out[-1])
    else:
      stitch_res.append(img1)

    stitch_indices.append(stitch_index_temp)
    fileLog.write("LEN:"+str(len(stitch_res)))
    #print(img2_idx)

  if(len(img2_idx) == 1):
    stitch_res.append(cv2.imread(filelist[img2_idx[0]]))
    stitch_indices.append(img2_idx)
    
  fileLog.write("LEN FINAL:"+str(len(stitch_res)))

  try:
    os.mkdir(os.path.join('/data1/naquib.alam/static/temp', filepath))
  except OSError as exc:
    try:
      os.mkdir('/data1/naquib.alam/static/temp')
      os.mkdir(os.path.join('/data1/naquib.alam/static/temp', filepath))
    except OSError as e:
      if e.errno != errno.EEXIST:
        raise
      pass

    if exc.errno != errno.EEXIST:
      raise
    pass


  # Save the results
  for i in range(len(stitch_res)):
    res_image_name = '/data1/naquib.alam/static/temp/' + filepath + '/result_' + str(i+1) + '.jpg'
    # res_image_nameC = 'static/temp/resultC_'+str(i)+'.JPG'
    # if stitch_res[i].shape[0] > 1200:
      # rows = 1200
      # cols = int(1200*stitch_res[i].shape[1]/stitch_res[i].shape[0])
    # else:
    try:
      os.mkdir(os.path.join(os.path.join('/data1/naquib.alam/static/temp', filepath), "result_"+str(i+1)))
    except OSError as exc:
      if exc.errno != errno.EEXIST:
        raise
      pass
    for index in stitch_indices[i]:
      path = os.path.join(os.path.join('/data1/naquib.alam/static/temp', filepath), "result_"+str(i+1))
      cv2.imwrite(os.path.join(path, str(index)+".jpg"), cv2.imread(filelist[index]))
    rows = stitch_res[i].shape[0]
    cols = stitch_res[i].shape[1]

    cv2.imwrite(res_image_name, stitch_res[i])
    fileLog.close()
    # cv2.imwrite(res_image_nameC, cv2.resize(stitch_res[i], (rows, cols)))

  # Save the inputs 
  # for f in filelist:
  #   res_image_nameC = 'static/images/'
  #   img = cv2.imread(f);
  #   if img.shape[0] > 1200:
  #     rows = 1200
  #     cols = int(1200*img.shape[1]/img.shape[0])
  #   else:
  #     rows = img.shape[0]
  #     cols = img.shape[1]

  #   cv2.imwrite(res_image_nameC + f[30:], cv2.resize(img, (rows, cols)))

  # for file in os.path.join("/data1/naquib.alam/images", filepath):
  #   if not os.path.isfile(os.path.join(os.path.join("/data1/naquib.alam/images", filepath), file)):
  #     continue
  #   os.remove(os.path.join(os.path.join("/data1/naquib.alam/images", filepath), file))
  # os.remove(os.path.join("/data1/naquib.alam/images", filepath))

  return stitch_indices

# Call main function
if __name__=='__main__':
  # print(sys.argv[1])
  main(sys.argv[1])
