import os, os.path
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from skimage import color, draw
import csv

def strandinator(arr, path = "./tiffs", valid_images = [".tiff"]):

    tiffs = [io.imread((os.path.join(path,f))) for f in os.listdir(path) if (os.path.splitext(f)[1]).lower() in valid_images]
        
    def separate(x, n):
        contoured = []
        ends = []
        cim = tiffs[x]
        for i in sorted(plt.contour(tiffs[x][:,:,3], levels = [1]).collections[0].get_paths(), key = lambda i : i.vertices.shape[0])[-n:]:
            vs = i.vertices
            r = vs[:,0]
            c = vs[:,1]
            rr, cc = draw.polygon(r,c)
            ends.append(np.max([cc,rr], axis = 1))
            contoured.append(np.zeros((max(cc)-min(cc)+1, max(rr)-min(rr)+1, 4), dtype = 'uint8'))
            contoured[-1][cc - min(cc), rr - min(rr)] = cim[cc, rr]
        return contoured, ends

    header = ['Original_Image', 'Strand', 'End', 'Resolution']

    with open('./Strands/Locations.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
    c = 0
    for i in range(len(tiffs)):
        contoured, ends = separate(i, arr[i])
        for j in range(len(contoured)):
            io.imsave('./Strands/imgs/' + str(c) + '.tiff', contoured[j])
            with open('./Strands/Locations.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow([i, c, ends[j], list(tiffs[i].shape[:2])])
            print(f'{i} - {c} saved')
            c += 1
