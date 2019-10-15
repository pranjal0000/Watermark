import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import os

mode = 'Test'

annotation_dir = '../Datasets/'+mode+'/annotation_files'
image_dir = '../Datasets/'+mode+'/Images'
mask_dir = '../Datasets/'+mode+'/Annotations'
img_size = (3648,5472,3)

mapping = {'Spot':0, 'Patch':1, 'Wrinkle':2}

for kev, val in mapping.items():
    if not os.path.exists(mask_dir+'/'+str(val)):
        os.mkdir(mask_dir+'/'+str(val))

for annotation_file in os.listdir(annotation_dir):
    #per annotation file
    if annotation_file == '.DS_Store':
        continue

    print("In annotation file:",annotation_file)

    with open(annotation_dir+'/'+annotation_file, 'r') as f:
        total_data = json.load(f)

    total = len(total_data)

    for i in range(total):
        #per image in an annotation
        cur_ann = total_data[i]
        image_name = cur_ann['filename'][-11:]

        # print(image_name)
        
        annot_all = cur_ann['annotations']
        data = {}
        # print(len(annot_all))

        for annotation in annot_all:
            atype = annotation['class']

            if not atype in data.keys():
                data[atype] = []

            x = ([float(x) for x in annotation['xn'].split(';')])
            y = ([float(y) for y in annotation['yn'].split(';')])

            toappend=[]
            for i in range(len(x)):
                toappend.append([x[i],y[i]])
            toappend = np.array(toappend,dtype=int)
            data[atype].append(toappend)
        # print(data)

        for key in data.keys():

            # ndata = np.array(data[key])
            # print(len(ndata),ndata.shape)

            img = np.zeros(img_size)
            # print(key)
            # img = plt.imread(image_dir+'/'+image_name) to overlay
            cv2.fillPoly(img, pts = data[key], color=(0,0,255))

            # print(img.shape)
            # plt.imshow(img)
            # plt.show()
            # print(img)
            save_name = mask_dir+'/'+str(mapping[key])+'/'+image_name[:-3]+'png'
            # print(save_name)
            # plt.imsave(save_name,img*255)
            cv2.imwrite(save_name, img)
            
            # print("Here",(img*255).shape)
            # print("Saved:",save_name)

# print(len(data))
# annot = data[0]['annotations']
# x=[]
# y=[]

# for spot in annot:
#   x.append([float(x) for x in spot['xn'].split(';')])
#   y.append([float(y) for y in spot['yn'].split(';')])

# # print(y[0])
# img = plt.imread(image_path)
# for i in range(len(x)):
#   plt.plot(x[i],y[i],linewidth=1)
# # plt.imsave('final.jpg',img)
# plt.plot(img)
# plt.show()