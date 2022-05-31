import os, os.path
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from skimage import color, draw
import csv

def datasetinator(flag = False, path = "./Strands/imgs"):
    
    valid_images = [".tiff"]
    imgs = [io.imread((os.path.join(path,f))) for f in os.listdir(path) if os.path.splitext(f)[1].lower() in valid_images]
    imgs_lab = [color.rgb2lab(imgs[i][:,:,:3]) for i in range(len(imgs))]
    
    def mask(x):
        blank0 = np.zeros(imgs[x].shape, dtype = 'uint8')
        cim = np.copy(imgs[x])
        for i in (plt.contour(np.where(np.logical_or(imgs_lab[x][:,:,1] >= 3, 
                                                     imgs_lab[x][:,:,1] == 0), 
                                       0, 255), levels = [1])).collections[0].get_paths():
            rr, cc = draw.polygon(i.vertices[:,0],i.vertices[:,1])
            if len(cc) > 0 and len(rr) > 0:
                blank0[cc, rr] = cim[cc, rr]
        return blank0
    
    def separate(x):
        contoured = []
        end = []
        cim = imgs[x]
        for i in (plt.contour(mask(x)[:,:,3], levels = [1])).collections[0].get_paths():
            vs = i.vertices
            r = vs[:,0]
            c = vs[:,1]
            rr, cc = draw.polygon(r,c)
            if len(cc) > 2:
                try:
                    end.append(np.max([cc,rr], axis = 1))
                except ValueError:
                    end.append([])
                temp = cim[cc, rr, :3]
                contoured.append(np.zeros((max(cc)-min(cc)+1, max(rr)-min(rr)+1, 4), dtype = 'uint8'))
                contoured[-1][cc - min(cc), rr - min(rr),:3] = temp
                contoured[-1][cc - min(cc), rr - min(rr), 3] = 255
        return contoured, end
    
    def clean(arr, arr2):
        i = 0
        while i < len(arr):
            if np.mean(arr[i]) < 10:
                arr.pop(i)
                arr2.pop(i)
            else:
                i += 1
        if len(arr) >= 2:

            maxarea = sorted(arr, key = lambda x : np.count_nonzero(x[:,:,3]))[-2]
            threshold = 0.1*maxarea.shape[0]*maxarea.shape[1]
            i = 0
            while i < len(arr):
                if (np.count_nonzero(arr[i][:,:,3]) < threshold) or max(arr[i].shape[:2])/min(arr[i].shape[:2]) > 3:
                    arr.pop(i)
                    arr2.pop(i)
                else:
                    i += 1
        return arr, arr2
    
    if flag:
        def testC(i):
            fig, ax = plt.subplots(ncols=1)
            sep1, sep2 = separate(i)
            contoured, end = clean(sep1, sep2)
            for c in end:
                ax.imshow(imgs[i])
                plt.plot(c[1],c[0], marker="o", markersize=7, markeredgecolor="red", markerfacecolor="green")
        for i in random.sample(range(len(imgs)), 4):
            testC(i)

    else:

        header = ['Strand', 'Filament', 'End', 'Resolution']

        with open('./Dataset/Locations.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
        c = 0
        for i in range(len(imgs)):
            sep1, sep2 = separate(i)
            contoured, end = clean(sep1, sep2)
            for j in range(len(contoured)):
                io.imsave('./Dataset/Filaments/' + str(c) + '.png', contoured[j])
                io.imsave('./Strands/imgs/' + str(i) + '.tiff', imgs[i])
                with open('./Dataset/Locations.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([i, c, end[j], list(imgs[i].shape[:2])])
                c += 1
            print(str(i) + " saved")
