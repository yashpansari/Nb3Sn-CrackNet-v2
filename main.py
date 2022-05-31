import os, os.path
import time
from skimage import io
from scipy.ndimage.interpolation import rotate
import numpy as np
from scipy.spatial import ConvexHull
from skimage import (color, io, draw, measure)
import csv
import pandas as pd
import random
import cv2
import matplotlib.pyplot as plt
import sys
from Strandinator import strandinator
from Datasetinator import datasetinator
from Crackinator import crackinator

def getStrand(i, images, df):
    index = df.loc[df["Strand"] == i]["Filament"]
    ends = df.loc[df["Strand"] == i]["End"]
    shapes = df.loc[df["Strand"] == i]["Resolution"]
    strand = []
    for i in index.to_list():
        strand.append(images[i])
    ends = np.array(ends.to_list())
    strand = np.array(strand)
    shapes = np.array(shapes.to_list())
    c = []
    for i in range(len(ends)):
        ends[i] = ends[i][1:len(ends[i]) - 1]
        c.append(ends[i].split(' '))
    for i in range(len(c)):
        c[i] = [x for x in c[i] if x != '']
        c[i] = [float(c[i][0]), float(c[i][1])]
    ends = [[int(x[0]), int(x[1])] for x in c]
    for i in range(len(shapes)):
        shapes[i] = shapes[i][1:len(shapes[i]) - 1]
        c.append(shapes[i].split(','))
    for i in range(len(c)):
        c[i] = [x for x in c[i] if x != '']
        c[i] = [float(c[i][0]), float(c[i][1])]
    shapes = [[int(x[0]), int(x[1])] for x in c]
    return shapes[-1], ends, strand

def construct(shape, ends, strand, cracked=[]):
    blank0 = np.zeros((shape[0], shape[1], 4), dtype='uint8')
    blank1 = np.zeros((shape[0], shape[1], 4), dtype='uint8')
    blank2 = np.zeros((shape[0], shape[1]), dtype='uint8')

    for i in range(len(strand)):
        h, w = strand[i].shape[:2]
        endx = ends[i][0]
        endy = ends[i][1]
        startx = endx - h
        starty = endy - w
        c = strand[i].copy()
        blank0[startx : endx, starty : endy] = np.maximum(c, blank0[startx : endx, starty : endy])
    avg_ones = np.count_nonzero(blank0[:,:,3])/len(strand)

    indices = blank0[:,:,3] > 0
    x = [i.any() for i in indices]
    for i in range(len(x)):
        if x[i]:
            startxx = i
            break
    for i in range(len(x) - 1, -1, -1):
        if x[i]:
            endxx = i
            break
    y = [i.any() for i in indices.T]
    for i in range(len(y)):
        if y[i]:
            startyy = i
            break
    for i in range(len(y) - 1, -1, -1):
        if y[i]:
            endyy = i
            break
            
    blank0 = blank0[startxx: endxx + 1, startyy: endyy + 1]

    blank3 = np.zeros(blank0.shape, dtype='uint8')

    blank1, blank2 = blank1[startxx: endxx + 1, startyy: endyy + 1], blank2[startxx: endxx + 1, startyy: endyy + 1]
    H, W = blank0.shape[:2]
    for i in range(len(strand)):
        if np.count_nonzero(strand[i][:,:,3]) >= avg_ones/7.5:
            h, w = strand[i].shape[:2]
            endx = ends[i][0] - startxx
            endy = ends[i][1] - startyy
            startx = endx - h
            starty = endy - w
            c = strand[i].copy()
            prob = 0.5 + 0.5*all([endx < H - 1, endy < W - 1, startx > 0, starty > 0])
            if np.max(cracked[i]):
                c[:,:,0] = 255*prob
                c[:,:,1] = 0
                c[:,:,2] = 0
                c[c[:,:,3]==0] = [0,0,0,0]
            else:
                c[:,:,0] = 0
                c[:,:,1] = 255*prob
                c[:,:,2] = 0
                c[c[:,:,3]==0] = [0,0,0,0]
            blank2[startx : endx, starty : endy] = np.maximum(cracked[i]*prob, blank2[startx : endx, starty : endy])
            blank1[startx : endx, starty : endy] = np.maximum(c, blank1[startx : endx, starty : endy])
            blank3[startx : endx, starty : endy] = np.maximum(strand[i]*prob, blank3[startx : endx, starty : endy])
    blank1[cv2.blur(blank2, (3,3)) > 0] = [0,0,0,0]
    blank3[cv2.blur(blank2, (13,13)) > 0] = [0,0,255,255]
    blank3[cv2.blur(blank2, (9,9)) > 0] = [0,255,0,255]
    blank3[cv2.blur(blank2, (5,5)) > 0] = [255,0,0,255]
    return blank0, blank1, blank3

def main(flag = sys.argv[1], show = sys.argv[2]):
    assert os.path.exists("./tiffs/") == True
    if flag == "True":
        os.makedirs("./Dataset/Filaments")
        os.makedirs("./Strands/imgs")
        strandinator(list(map(int, sys.argv[3:])))
        datasetinator()
        crackinator()
    if show == "True":
        print("showing...")
        path = "./Dataset/Filaments"
        valid_images = [".jpg",".gif",".png",".tga"]
        imgs = [io.imread((os.path.join(path,f))) for f in os.listdir(path) if (os.path.splitext(f)[1]).lower() in valid_images]
        path = "./Dataset/Cracks"
        cracks = [io.imread((os.path.join(path,f))) for f in os.listdir(path) if (os.path.splitext(f)[1]).lower() in valid_images]
        imgs_gray = [color.rgb2gray(imgs[i]) for i in range(len(imgs))]
        df = pd.read_csv("./Dataset/Locations.csv")
        constructs = []
        marked_constructs = []
        cracked_constructs = []
        for i in range(df['Strand'].nunique()):
            try:
                shape, ends, strand = getStrand(i, imgs, df)
                shape2, ends2, strand2 = getStrand(i, cracks, df)
                a, b, c = construct(shape, ends, strand, strand2)
                constructs.append(np.flipud(io.imread("./Strands/imgs/"+str(i)+".tiff")))
                cracked_constructs.append(b)
                marked_constructs.append(c)
            except IndexError:
                continue
        for i in range(len(constructs)):
            fig, ax = plt.subplots(ncols = 3, figsize = (100,500))
            ax[0].imshow(constructs[i][:,:,:3])
            ax[1].imshow(marked_constructs[i][:,:,:3])
            ax[2].imshow(cracked_constructs[i][:,:,:3])
            plt.show()

if __name__ == "__main__":
    main()