import os, os.path
from scipy.ndimage.interpolation import rotate
import numpy as np
from scipy.spatial import ConvexHull
from skimage import (color, io, draw, measure)
import csv
import cv2
import matplotlib.pyplot as plt

def crackinator(path = "./Dataset/Filaments"):
  valid_images = [".jpg",".gif",".png",".tga", ".tiff"]
  imgs = [io.imread((os.path.join(path,f))) for f in os.listdir(path) if (os.path.splitext(f)[1]).lower() in valid_images]
  len(imgs)
  imgs_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]
  def canny(x):
    img = imgs[x].copy()
    img_gray = imgs_gray[x]
    img_hsv = cv2.cvtColor(img[:,:,:3], cv2.COLOR_RGB2HSV)[:,:,1]
    med = np.median(img_gray[img_gray!=0])
    edges = cv2.Canny(image=cv2.GaussianBlur(img_gray, (3,3), 0), threshold1=0.05*np.median(img_gray[img_gray!=0]), threshold2=int(min(255,0.4*med)))
    med = np.median(img_hsv[img_gray!=0])
    edges2 = cv2.Canny(image=cv2.blur(img_hsv, (2,2), 0), threshold1=min(90, 10*med), threshold2=min(120,32*med))
    for p in (plt.contour(imgs_gray[x], levels = [1])).collections[0].get_paths():
      vs = p.vertices
      vs = np.array([[round(v[0]), round(v[1])] for v in vs])
      cc, rr = [],[]
      for i in range(len(vs) - 1):
          r, c = draw.line(vs[i, 0], vs[i, 1], vs[i+1, 0], vs[i+1, 1])
          rr += list(r)
          cc += list(c)
      edges2[cc,rr] = 255
    edges2 = cv2.blur(edges2, (7,7))
    edges[edges2 != 0] = 0

    return edges

  def minimum_bounding_rectangle(points):
    
      pi2 = np.pi/2.

      # get the convex hull for the points
      hull_points = points[ConvexHull(points, qhull_options='QJ').vertices]

      # calculate edge angles
      edges = np.zeros((len(hull_points)-1, 2))
      edges = hull_points[1:] - hull_points[:-1]

      angles = np.zeros((len(edges)))
      angles = np.arctan2(edges[:, 1], edges[:, 0])

      angles = np.abs(np.mod(angles, pi2))
      angles = np.unique(angles)

      # find rotation matrices
      rotations = np.vstack([
          np.cos(angles),
          np.cos(angles-pi2),
          np.cos(angles+pi2),
          np.cos(angles)]).T

      rotations = rotations.reshape((-1, 2, 2))

      # apply rotations to the hull
      rot_points = np.dot(rotations, hull_points.T)

      # find the bounding points
      min_x = np.nanmin(rot_points[:, 0], axis=1)
      max_x = np.nanmax(rot_points[:, 0], axis=1)
      min_y = np.nanmin(rot_points[:, 1], axis=1)
      max_y = np.nanmax(rot_points[:, 1], axis=1)

      # find the box with the best area
      areas = (max_x - min_x) * (max_y - min_y)
      best_idx = np.argmin(areas)

      # return the best box
      x1 = max_x[best_idx]
      x2 = min_x[best_idx]
      y1 = max_y[best_idx]
      y2 = min_y[best_idx]
      r = rotations[best_idx]

      rval = np.zeros((4, 2))
      rval[0] = np.dot([x1, y2], r)
      rval[1] = np.dot([x2, y2], r)
      rval[2] = np.dot([x2, y1], r)
      rval[3] = np.dot([x1, y1], r)

      return rval

  edges = [canny(x) for x in range(len(imgs))]
  qcs = [plt.contour(edge, levels = [np.median(edge[edge!=0])]) for edge in edges]

  def rgb2lab(inputColor):

      num = 0
      RGB = [0, 0, 0]

      for value in inputColor:
          value = float(value) / 255

          if value > 0.04045:
              value = ((value + 0.055) / 1.055) ** 2.4
          else:
              value = value / 12.92

          RGB[num] = value * 100
          num = num + 1

      XYZ = [0, 0, 0, ]

      X = RGB[0] * 0.4124 + RGB[1] * 0.3576 + RGB[2] * 0.1805
      Y = RGB[0] * 0.2126 + RGB[1] * 0.7152 + RGB[2] * 0.0722
      Z = RGB[0] * 0.0193 + RGB[1] * 0.1192 + RGB[2] * 0.9505
      XYZ[0] = round(X, 4)
      XYZ[1] = round(Y, 4)
      XYZ[2] = round(Z, 4)

      # Observer= 2°, Illuminant= D65
      XYZ[0] = float(XYZ[0]) / 95.047         # ref_X =  95.047
      XYZ[1] = float(XYZ[1]) / 100.0          # ref_Y = 100.000
      XYZ[2] = float(XYZ[2]) / 108.883        # ref_Z = 108.883

      num = 0
      for value in XYZ:

          if value > 0.008856:
              value = value ** (0.3333333333333333)
          else:
              value = (7.787 * value) + (16 / 116)

          XYZ[num] = value
          num = num + 1

      Lab = np.array([0, 0, 0])

      L = (116 * XYZ[1]) - 16
      a = 500 * (XYZ[0] - XYZ[1])
      b = 200 * (XYZ[1] - XYZ[2])

      Lab[0] = round(L, 4)
      Lab[1] = round(a, 4)
      Lab[2] = round(b, 4)

      return Lab

  def rejection(x, box, major, test = False):
    h, w = imgs[x].shape[0], imgs[x].shape[1] #dimensions

    side1 = max([[box[0], box[1]], [box[1], box[2]]], 
               key = lambda s : np.linalg.norm(s[0] - s[1])) #left major axis
    point = np.array(box[0])/2 + np.array(box[2])/2 #centroid
    move = point - np.add(side1[0], side1[1])/2 #vector from 
    side = [np.add(side1[0], move), np.add(side1[1], move)]  #Central Major axis
    side2 = [np.add(side1[0], 2*move), np.add(side1[1], 2*move)] #right major axis

    mid1 = np.array(side1[0])/2 + np.array(side1[1])/2 - 5*move/np.linalg.norm(move)
    mid2 = np.array(side2[0])/2 + np.array(side2[1])/2 + 5*move/np.linalg.norm(move)   
    mid1 = np.maximum(0, np.minimum(mid1, [w - 1, h - 1])) #left major axis midpoint
    mid2 = np.maximum(0, np.minimum(mid2, [w - 1, h - 1])) #right major axis midpoint
    
    borders = [np.array([np.array(list(map(int, side[0] - 1))), np.array(list(map(int,side[1] - 1)))])
               for i in range(4)] #filament border segments [left, right, bottom, top]
    
    for i in range(1):
          
      border = borders[0] #left
      try:
          while border[0][0] > 0 and imgs[x][border[0][1], border[0][0], 3]:
              border[0][0] -= 1
      except:
          border[0][0] = w - 1
      try:
          while border[1][0] > 0 and imgs[x][border[1][1], border[1][0], 3]:
              border[1][0] -= 1
      except:
          border[1][0] = w - 1
          
      border = borders[1] #right
      try:
        while border[0][0] < w and imgs[x][border[0][1], border[0][0], 3]:
          border[0][0] += 1
      except:
          border[0][0] = w - 1
      try:
        while border[1][0] < w and imgs[x][border[1][1], border[1][0], 3]:
          border[1][0] += 1
      except:
        border[0][0], border[1][0] = w - 1, w - 1

      border = borders[2] #bottom
      try:
          while border[0][1] > 0 and imgs[x][border[0][1], border[0][0], 3]:
              border[0][1] -= 1
      except:
          border[0][1] = h - 1
      try:
          while border[1][1] > 0 and imgs[x][border[1][1], border[1][0], 3]:
              border[1][1] -= 1
      except:
          border[1][1] = h - 1

      border = borders[3] #top
      try:
          while border[0][1] < h and imgs[x][border[0][1], border[0][0], 3]:
              border[0][1] += 1
      except:
          border[0][1] = h - 1
      try:
          while border[1][1] < h and imgs[x][border[1][1], border[1][0], 3]:
            border[1][1] += 1
      except:
        border[1][1] = h - 1
    
    border = min(borders, key = lambda b : np.linalg.norm((b[0] + b[1])/2 - point) 
                 if (b>0).all() else float('inf')) #closest border that isn't cut off

    #angle calculation in degrees
    vector_1 = [side[0][1] - side[1][1], side[0][0] - side[1][0]]
    vector_2 = [border[0][1] - border[1][1], border[0][0] - border[1][0]] 
    dot_product = np.dot(vector_1, vector_2)/(np.linalg.norm(vector_1)*np.linalg.norm(vector_2))
    angle = np.arccos(dot_product)
    angle = min(angle, np.pi - angle)*180/np.pi
    
    #color difference calculator
    diff = float('inf')
    try:
          color1 = rgb2lab(imgs[x][round(mid1[1]), round(mid1[0]), :3])
          color2 = rgb2lab(imgs[x][round(mid2[1]), round(mid2[0]), :3])
          diff = np.linalg.norm(color2 - color1)
    except: #nan Lab values
          pass

      
    #plotting to check
    if test:
        fig, ax = plt.subplots()
        ax.imshow(imgs[x])
        ax.plot([side[0][0], side[1][0]],[side[0][1], side[1][1]], '-g')
        ax.plot([border[0][0], border[1][0]],[border[0][1], border[1][1]], '-r')
        ax.plot(mid1[0], mid1[1], '-bo')
        ax.plot(mid2[0], mid2[1], '-wo')
        ax.plot(np.append(box[:,0], box[0,0]), np.append(box[:,1], box[0,1]))
        ax.set_title("Angle: " + str(angle) + 
                     #", Size: " + str(ratio) +
                     ", Diff:" + str(diff) + 
                     ", Color: " + ", ".join(list(map(str,color1))) + "; " + ", ".join(list(map(str,color2))))

    return bool(angle <= 25 #angle between major axis and boundary
                or (diff >= 9 and max(color1[2], color2[2]) >= 4) #color diff > 9 and one yellow/red
                or diff >= 18 #color diff just too high
                or max(color1[2], color2[2]) > 5 #any yellow/red
                or abs(color2[-1] - color1[-1]) > 6
                or min(color1[0], color2[0]) <= 55) #any too dark

  def cracks(x, test = False):
    abcd = []
    new = np.zeros(edges[x].shape, dtype = 'uint8')
    arr = np.nonzero(imgs_gray[x])
    if len(arr[0]) < 3:
      return new, abcd
    box = minimum_bounding_rectangle(np.column_stack(arr))
    l = sorted((np.linalg.norm(box[1] - box[0]), np.linalg.norm(box[2] - box[1])))
    for p in qcs[x].collections[0].get_paths():
      vs = p.vertices
      r = vs[:,0]
      c = vs[:,1]
      rr, cc = draw.polygon(r,c)
      if len(rr) < 3:
        continue
      box = minimum_bounding_rectangle(np.column_stack((rr,cc)))
      lengths = sorted((np.linalg.norm(box[1] - box[0]), np.linalg.norm(box[2] - box[1])))
      if 7.5 < l[0]/lengths[1] or lengths[1]/lengths[0] < 1.8 or rejection(x, box, lengths[1], test):
        continue
      abcd.append([np.append(box[:,0], box[0,0]), np.append(box[:,1], box[0,1])])
      new[cc, rr] = edges[x][cc, rr]
      
    return new, abcd

  for x in range(total):
    io.imsave(f'./Dataset/Cracks/{str(x)}.png', cracks(x)[0])
