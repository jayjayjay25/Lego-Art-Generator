import pandas as pd
from PIL import Image
import numpy as np
import sklearn.cluster as c
import os
from pathlib import Path


def make_pic(filename,dim):
    img = Image.open(filename)
    img = img.convert('RGB')
    if dim[0]!=0 and dim[1]!= 0:
        img = img.resize(dim)
    img.show()
    img_mat = np.array(img)
    img_df = np.zeros((img_mat.shape[1]*img_mat.shape[0],5))
    print(img_mat.shape)
    img_df = pd.DataFrame(img_df,columns= ["R","G","B","row","column"])
    count = 0
    for i in range(img_mat.shape[0]):
        for j in range(img_mat.shape[1]):
            img_df["R"][count] = img_mat[i][j][0]
            #print(img_mat[i][j][0])
            img_df["G"][count] = img_mat[i][j][1]
            img_df["B"][count] = img_mat[i][j][2]
            img_df["row"][count] = i
            img_df["column"][count] = j
            count = count + 1
    img_df = pd.DataFrame(img_df)
    image_df = img_df.drop(columns = ["row","column"])
    return (img_df,image_df,img,img_mat.shape)



df = pd.DataFrame(columns = ["R","G","B"])
#filename = 'melvin.jpeg'
filename = 'time_dali.jpeg'
#filename = 'ditto.png'
#filename = 'lego_test_monroe.png'
#filename = 'the_scream.jpeg'
#filename = 'saturn.jpg'
#filename = 'van_gogh.jpg'
#filename = 'van_gogh_flowers.jpg'
#filename = 'a_sunday_on_la_grande.jpg'
#filename = 'nighthawks.jpg'
#filename = 'pink_floyd.jpg'
#filename = 'crimson_king.jpg'
#filename = 'david.jpg'

dim = (53,53) #I use (53,53) because these are the dimensions they use in the lego builds, however you can scale these to whatever you want
         # if it means capturing more information from the picture. Set it to (0,0) for the actual image dimensions
(img_df,image_df,img,dims) = make_pic(filename,dim)
fileage = open("bricklink_colors.csv","r")
lcolors_df = pd.read_csv(fileage)

#kmeans to find regions/clusters
k = 18 #set this to however many colors you think appear in the image. More complex images will need more clusters, but simpler images will likely
       # suffer from greater than about 15 clusters/colors
kmeans = c.KMeans(n_clusters=k).fit(image_df)
label_vals = kmeans.cluster_centers_
best_cols = []
for label in label_vals: #for each centroid, we choose the lego color closest to it for k colors associated with k clusters
     best_cols.append(lcolors_df.iloc[:,np.argmin(np.array([ np.dot(lcolors_df[color]-label,lcolors_df[color]-label) for color in lcolors_df]))])
best_col_names = pd.DataFrame(best_cols).transpose()

color_counter = np.zeros(len(best_cols))
for i in range(len(image_df)):
    image_df.iloc[i] = best_cols[kmeans.labels_[i]]
    color_counter[kmeans.labels_[i]]+=1
new_img_df = image_df.join(img_df["row"]).join(img_df["column"])
print(new_img_df)
new_img_mat = np.zeros((dims[0],dims[1],3))
for i in range(len(new_img_df)): #whichever cluster each pixel belongs to determines the lego color to which it is assigned
    new_img_mat[int(new_img_df["row"].iloc[i])][int(new_img_df["column"].iloc[i])] = [new_img_df["R"].iloc[i],new_img_df["G"].iloc[i],new_img_df["B"].iloc[i]]
new_img = Image.fromarray(new_img_mat.astype('uint8'), mode='RGB')
new_img.show()
tmp = [best_col_names.columns.tolist(),color_counter]
colors_to_get = pd.DataFrame({"Color Name" : best_col_names.columns.tolist(), "Quantity" : color_counter})
print(colors_to_get)

#filepath = Path('colors_to_get.csv')  
#filepath.parent.mkdir(parents=True, exist_ok=True)  
#colors_to_get.to_csv(filepath, index=False) 
