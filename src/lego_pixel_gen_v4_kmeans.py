import warnings
import pandas as pd
import numpy as np
import itertools as itz
import plotly.express as px
import plotly.graph_objects as pxg
import statistics as st
import random as ran
from PIL import Image
import math
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
# Import ElbowVisualizer
import sklearn as sk
from sklearn.cluster import KMeans
import yellowbrick as yb
from yellowbrick.cluster import KElbowVisualizer
##from IPython.display import display
#import seaborn as sn

def most_common_used_color(img,filename):
    # Get width and height of Image
    width, height = img.size
 
    # Initialize Variable
    r_total = 0
    g_total = 0
    b_total = 0
 
    count = 0
 
    # Iterate through each pixel
    for x in range(0, width):
        for y in range(0, height):
            # r,g,b value of pixel
            (r, g, b) = img.getpixel((x, y))
 
            r_total += r
            g_total += g
            b_total += b
            count += 1
 
    return (filename,r_total/count, g_total/count, b_total/count)
def find_values(filename: str):
    img = Image.open(filename)
    img = img.convert('RGB')
    newsize = (1,1)
    img = img.resize(newsize)
    #img_mat = np.asarray(img)
    #print(img_mat)
    tuppy = most_common_used_color(img,filename)
    #print(filename)
    #print(tuppy)
    return tuppy

def hsl(r,g,b):
    r = float(r)
    g = float(r)
    b = float(r)
    set = [r,g,b]
    x = min(set)
    y = max(set)
    lum = (min(set)+max(set))/2
    if x==y:
        return (lum,0,0)
    else:
        if lum <= 0.5:
            sat = (y-x)/(x+y)
        else:
            sat = (y-x)/(2-y-x)
        if y==r:
            hue = (g-b)/(y-x)
        if y==g:
            hue = 2 + (b-r)/(y-x)
        if y==b:
            hue = 4+ (r-g)/(y-x)
        hue = hue*60
        if hue < 0:
            hue = hue + 360
        return (lum,sat,hue)
    

def error_finding(img, img_mat, new_df,i,j,l):
            value1 = math.sqrt((img_mat[i][j][0] - new_df[l][0])**2 + (img_mat[i][j][1] - new_df[l][1])**2 + (img_mat[i][j][2] - new_df[l][2])**2)
            #value2 =  math.sqrt( (math.sqrt((img_mat[i][j][0]**2) + (img_mat[i][j][1])**2 + (img_mat[i][j][2])**2) - math.sqrt((new_df[l][0])**2 + (new_df[l][1])**2 + (new_df[l][2])**2))**2 )

            #comparing with surrounding pixels
            #string_holder = """
            values_included = 0
            if (i > 0) & (i < img.height-1):
                value2 = math.sqrt((img_mat[i+1][j][0] - new_df[l][0])**2 + (img_mat[i+1][j][1] - new_df[l][1])**2 + (img_mat[i+1][j][2] - new_df[l][2])**2) + math.sqrt((img_mat[i-1][j][0] - new_df[l][0])**2 + (img_mat[i-1][j][1] - new_df[l][1])**2 + (img_mat[i-1][j][2] - new_df[l][2])**2)
                values_included = values_included + 2
            elif (i == 0) & (i < img.height-1):
                value2 = math.sqrt((img_mat[i+1][j][0] - new_df[l][0])**2 + (img_mat[i+1][j][1] - new_df[l][1])**2 + (img_mat[i+1][j][2] - new_df[l][2])**2)
                values_included = values_included + 1
            elif (i > 0) & (i == img.height-1):
                value2 = math.sqrt((img_mat[i-1][j][0] - new_df[l][0])**2 + (img_mat[i-1][j][1] - new_df[l][1])**2 + (img_mat[i-1][j][2] - new_df[l][2])**2)
                values_included = values_included + 1

            if (j > 0) & (j < img.width-1):
                value3 = math.sqrt((img_mat[i][j+1][0] - new_df[l][0])**2 + (img_mat[i][j+1][1] - new_df[l][1])**2 + (img_mat[i][j+1][2] - new_df[l][2])**2) + math.sqrt((img_mat[i][j-1][0] - new_df[l][0])**2 + (img_mat[i][j-1][1] - new_df[l][1])**2 + (img_mat[i][j-1][2] - new_df[l][2])**2)
                values_included = values_included + 2
            elif (j == 0) & (j < img.width-1):
                value3 = math.sqrt((img_mat[i][j+1][0] - new_df[l][0])**2 + (img_mat[i][j+1][1] - new_df[l][1])**2 + (img_mat[i][j+1][2] - new_df[l][2])**2)
                values_included = values_included + 1
            elif (j > 0) & (j == img.width-1):
                value3 = math.sqrt((img_mat[i][j-1][0] - new_df[l][0])**2 + (img_mat[i][j-1][1] - new_df[l][1])**2 + (img_mat[i][j-1][2] - new_df[l][2])**2)
                values_included = values_included + 1

            if (i > 0) & (i < img.height-1) & (j > 0) & (j < img.width-1):
                value4 = math.sqrt((img_mat[i-1][j-1][0] - new_df[l][0])**2 + (img_mat[i-1][j-1][1] - new_df[l][1])**2 + (img_mat[i-1][j-1][2] - new_df[l][2])**2)
                + math.sqrt((img_mat[i+1][j-1][0] - new_df[l][0])**2 + (img_mat[i+1][j-1][1] - new_df[l][1])**2 + (img_mat[i+1][j-1][2] - new_df[l][2])**2)
                + math.sqrt((img_mat[i+1][j+1][0] - new_df[l][0])**2 + (img_mat[i+1][j+1][1] - new_df[l][1])**2 + (img_mat[i+1][j+1][2] - new_df[l][2])**2)
                + math.sqrt((img_mat[i-1][j+1][0] - new_df[l][0])**2 + (img_mat[i-1][j+1][1] - new_df[l][1])**2 + (img_mat[i-1][j+1][2] - new_df[l][2])**2)
                values_included = values_included + 4            
            elif (i > 0) & (i < img.height-1) & (j == img.width-1):
                value4 = math.sqrt((img_mat[i+1][j-1][0] - new_df[l][0])**2 + (img_mat[i+1][j-1][1] - new_df[l][1])**2 + (img_mat[i+1][j-1][2] - new_df[l][2])**2)
                + math.sqrt((img_mat[i-1][j-1][0] - new_df[l][0])**2 + (img_mat[i-1][j-1][1] - new_df[l][1])**2 + (img_mat[i-1][j-1][2] - new_df[l][2])**2)
                values_included = values_included + 2
            elif (i > 0) & (i < img.height-1) & (j==0):
                value4 = math.sqrt((img_mat[i-1][j+1][0] - new_df[l][0])**2 + (img_mat[i-1][j+1][1] - new_df[l][1])**2 + (img_mat[i-1][j+1][2] - new_df[l][2])**2)
                + math.sqrt((img_mat[i+1][j+1][0] - new_df[l][0])**2 + (img_mat[i+1][j+1][1] - new_df[l][1])**2 + (img_mat[i+1][j+1][2] - new_df[l][2])**2)
                values_included = values_included + 2
            elif (i==0) & (j > 0) & (j < img.width-1):
                value4 = math.sqrt((img_mat[i+1][j-1][0] - new_df[l][0])**2 + (img_mat[i+1][j-1][1] - new_df[l][1])**2 + (img_mat[i+1][j-1][2] - new_df[l][2])**2)
                + math.sqrt((img_mat[i+1][j+1][0] - new_df[l][0])**2 + (img_mat[i+1][j+1][1] - new_df[l][1])**2 + (img_mat[i+1][j+1][2] - new_df[l][2])**2)
                values_included = values_included + 2
            elif (i == img.height-1) & (j > 0) & (j < img.width-1):
                value4 = math.sqrt((img_mat[i-1][j+1][0] - new_df[l][0])**2 + (img_mat[i-1][j+1][1] - new_df[l][1])**2 + (img_mat[i-1][j+1][2] - new_df[l][2])**2)
                + math.sqrt((img_mat[i-1][j-1][0] - new_df[l][0])**2 + (img_mat[i-1][j-1][1] - new_df[l][1])**2 + (img_mat[i-1][j-1][2] - new_df[l][2])**2)
                values_included = values_included + 2
            elif (i == 0) & (j == 0):
                value4 = math.sqrt((img_mat[i+1][j+1][0] - new_df[l][0])**2 + (img_mat[i+1][j+1][1] - new_df[l][1])**2 + (img_mat[i+1][j+1][2] - new_df[l][2])**2)
                values_included = values_included + 1
            elif (i == 0) & (j == img.width-1):
                value4 = math.sqrt((img_mat[i+1][j-1][0] - new_df[l][0])**2 + (img_mat[i+1][j-1][1] - new_df[l][1])**2 + (img_mat[i+1][j-1][2] - new_df[l][2])**2)
                values_included = values_included + 1
            elif (i == img.height-1) & (j == img.width-1):
                value4 = math.sqrt((img_mat[i-1][j-1][0] - new_df[l][0])**2 + (img_mat[i-1][j-1][1] - new_df[l][1])**2 + (img_mat[i-1][j-1][2] - new_df[l][2])**2)
                values_included = values_included + 1
            elif (i == img.height-1) & (j == 0):
                value4 = math.sqrt((img_mat[i-1][j+1][0] - new_df[l][0])**2 + (img_mat[i-1][j+1][1] - new_df[l][1])**2 + (img_mat[i-1][j+1][2] - new_df[l][2])**2)
                values_included = values_included + 1

            #value = value1 + value2
            value = value1 + (value2 + value3 + value4)/values_included
            return value

def kmeans(df: pd.DataFrame, cluster_count: int, list_of_clusters: list):#,list_of_indices):#list of clusters should be a list of dim 3 tuples (or "VOID")
    dee_ef = df.drop(columns = ["row","column"])
    dee_ef = dee_ef.transpose()
    list_of_names = list(dee_ef.columns.values)
    stuff = ran.sample(list_of_names, cluster_count)
    for i in range (0,len(list_of_clusters)):
        if list_of_clusters[i] != "VOID":
            stuff[i] = list_of_clusters[i]
    df_stuff = dee_ef[stuff]
    cluster_df = np.zeros((len(df),cluster_count))
    cluster_df = pd.DataFrame(cluster_df)
    print(cluster_df)
    distlist = []
    mindex =0
    for columns in dee_ef:
        for k in df_stuff:
            pixelQ = df[columns]
            pixelC = df_stuff[k]
            dist = (pixelQ[0] - pixelC[0])**2 + (pixelQ[1] - pixelC[1])**2 + (pixelQ[2] - pixelC[2])**2
            distlist.append(dist)
        for i in range(0,len(distlist)):
            if min(distlist) == distlist[i]:
                mindex = i
            cluster_df[mindex][columns] = 1
        distlist = []
    print(cluster_df)
    sum = [0,0,0]
    dee_ef = np.array(dee_ef)
    print("bruh: ")
    print(dee_ef)
    avgs = []
    for i in range(0,len(cluster_count)):
        sum = [0,0,0]
        for j in range(0,len(df)):
            if cluster_df[j][i]==1:
                sum[0]=dee_ef[0][j]
                sum[1]=dee_ef[1][j]
                sum[2]=dee_ef[2][j]
                print("moment: ")
                print(dee_ef[2][j])
        length_cluster = sum/cluster_df[i].sum()
        avgs.append(length_cluster)
    print(avgs)
    return(cluster_count,avgs,cluster_df)#,indices)
#extracting color possibilities
def make_colors():
    data = []
    k = find_values('black_plate.png')
    data.append(k)
    k = find_values('brightyellow_plate.png')
    data.append(k)
    k = find_values('chromegold_plate.png')
    data.append(k)
    k = find_values('transparentyellow_plate.png')
    data.append(k)
    k = find_values('yellow_plate.png')
    data.append(k)
    k = find_values('flatdarkgold_plate.png')
    data.append(k)
    k = find_values('darkred_plate.png')
    data.append(k)
    k = find_values('transparentred_plate.png')
    data.append(k)
    k = find_values('transparentlightorange_plate.png')
    data.append(k)
    k = find_values('transparentreddishorange_plate.png')
    data.append(k)
    k = find_values('brightlightorange_plate.png')
    data.append(k)
    k = find_values('transparentorange_plate.png')
    data.append(k)
    k = find_values('orange_plate.png')
    data.append(k)
    k = find_values('darkorange_plate.png')
    data.append(k)
    k = find_values('pearlgold_plate.png')
    data.append(k)
    k = find_values('lightgreen_plate.png')
    data.append(k)
    k = find_values('translucentlightbrightgreen_plate.png')
    data.append(k)
    k = find_values('transparentneongreen_plate.png')
    data.append(k)
    k = find_values('green_plate.png')
    data.append(k)
    k = find_values('lightlime_plate.png')
    data.append(k)
    k = find_values('transparentlimegreen_plate.png')
    data.append(k)
    k = find_values('lime_plate.png')
    data.append(k)
    k = find_values('transparentgreen_plate.png')
    data.append(k)
    k = find_values('sandgreen_plate.png')
    data.append(k)
    k = find_values('brightgreen_plate.png')
    data.append(k)
    k = find_values('transparentdarkblue_plate.png')
    data.append(k)
    k = find_values('transparentmediumblue_plate.png')
    data.append(k)
    k = find_values('mediumazure_plate.png')
    data.append(k)
    k = find_values('blue_plate.png')
    data.append(k)
    k = find_values('sandblue_plate.png')
    data.append(k)
    k = find_values('darkblue_plate.png')
    data.append(k)
    k = find_values('lightaqua_plate.png')
    data.append(k)
    k = find_values('mediumblue_plate.png')
    data.append(k)
    k = find_values('transparentverylightblue_plate.png')
    data.append(k)
    k = find_values('violet_plate.png')
    data.append(k)
    k = find_values('maerskblue_plate.png')
    data.append(k)
    k = find_values('transparentpink_plate.png')
    data.append(k)
    k = find_values('magenta_plate.png')
    data.append(k)
    k = find_values('mediumdarkpink_plate.png')
    data.append(k)
    k = find_values('brightpink_plate.png')
    data.append(k)
    k = find_values('pink_plate.png')
    data.append(k)
    k = find_values('transparentdarkpink_plate.png')
    data.append(k)
    k = find_values('transparentpink_plate.png')
    data.append(k)
    k = find_values('darkpink_plate.png')
    data.append(k)
    k = find_values('mediumlavender_plate.png')
    data.append(k)
    k = find_values('purple_plate.png')
    data.append(k)
    k = find_values('darkpurple_plate.png')
    data.append(k)
    k = find_values('transparentpurple_plate.png')
    data.append(k)
    k = find_values('darkgray_plate.png')
    data.append(k)
    k = find_values('metallicsilver_plate.png')
    data.append(k)
    k = find_values('pearllightgray_plate.png')
    data.append(k)
    k = find_values('chromesilver_plate.png')
    data.append(k)
    k = find_values('darkstonegray_plate.png')
    data.append(k)
    k = find_values('flatgray_plate.png')
    data.append(k)
    k = find_values('pearldarkgray_plate.png')
    data.append(k)
    k = find_values('transparent_plate.png')
    data.append(k)
    k = find_values('mediumstonegray_plate.png')
    data.append(k)
    k = find_values('transparentblack_plate.png')
    data.append(k)
    k = find_values('reddishbrown_plate.png')
    data.append(k)
    k = find_values('warmtan_plate.png')
    data.append(k)
    k = find_values('flesh_plate.png')
    data.append(k)
    k = find_values('darktan_plate.png')
    data.append(k)
    k = find_values('brown_plate.png')
    data.append(k)
    k = find_values('translucentwhite_plate.png')
    data.append(k)
    k = find_values('transparentopal_plate.png')
    data.append(k)
    k = find_values('modulexwhite_plate.png')
    data.append(k)
    k = find_values('darkturquoise_plate.png')
    data.append(k)
    k = find_values('yellowishgreen_plate.png')
    data.append(k)
    k = find_values('tan_plate.png')
    data.append(k)
    k = find_values('olivegreen_plate.png')
    data.append(k)
    k = find_values('mediumdarkflesh_plate.png')
    data.append(k)
    k = find_values('red_plate.png')
    data.append(k)
    return data
   
#creating the set of lego colors df
data = make_colors()

df = pd.DataFrame(data, columns = ['color names','Red value','Green value','Blue value'])
print(df)
color_names = df['color names']
df = df.drop(columns = ['color names'])
df = df.astype(float)
new_df = df.to_numpy()
#opening the image
#img = Image.open('The_Garden_of_earthly_delights.jpeg')
#img = Image.open('melvin.jpeg')
img = Image.open('ditto.jpeg')
#img = Image.open('lego_test_monroe.png')
img = img.convert('RGB')
newsize = (53,53)
img = img.resize(newsize)
img.show()
img_mat = np.array(img)

img_df = []

img_df = np.zeros((len(img_mat)**2,5))
img_df = pd.DataFrame(img_df,columns= ["R","G","B","row","column"])
print(img_df)
count = 0
for i in range(0,len(img_mat)):
    for j in range(0,len(img_mat)):
        img_df["R"][count] = img_mat[i][j][0]
        #print(img_mat[i][j][0])
        img_df["G"][count] = img_mat[i][j][1]
        img_df["B"][count] = img_mat[i][j][2]
        img_df["row"][count] = i
        img_df["column"][count] = j
        count = count + 1
#img_df = pd.DataFrame(img_df)
print(img_df)
image_df = img_df.drop(columns = ["row","column"])
model = KMeans()
# k is range of number of clusters.
visualizer = KElbowVisualizer(model, k=(4,70), timings=False)
visualizer.fit(img_df)        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure

list_of_clusters = []
list_of_clusters.append("VOID")
#for i in range(0,50):
(cluster_count,avgs,cluster_df) = kmeans(img_df,24,list_of_clusters)
j=0
while j != 1:
    count = 0
    (new_cc,new_avgs,new_cdf) = kmeans(img_df,24,avgs)
    #for i in range(0,24):
    #    if new_avgs[i] == avgs[i]:
    #        count = count + 1
    if new_cdf==cluster_df:
        print("done!")
        j = 1
    else:
        cluster_count = new_cc
        avgs = new_avgs
mindist = 10000000000000
mincol = [33,33,33]
corr_colors = []
corr_color_names = []
color_count = np.zeros([len(new_df)])
mindex = 0
for i in range(0,len(avgs)):
    for j in range(0,new_df):
        dist = (new_df[j][0] - new_avgs[j][0])**2 + (new_df[j][1] - new_avgs[i][1])**2 + (new_df[j][2] - new_avgs[i][2])**2
        if dist < mindist:
            mindist = dist
            mincol = new_df[j]
            mindex = color_names[j]
    corr_colors.append(mincol)
    corr_color_names.append(mindex)
for j in range(0,24): #replacing each color in the image with 
    for i in range (0,len(img_df)):   
        if new_cdf[i][j] == 1:
            #img_df["R"].iloc[i] = mincol[j][0]
            #img_df["G"].iloc[i] = mincol[j][1]
            #img_df["B"].iloc[i] = mincol[j][2]
            img_mat[img_df["row"].iloc[i]][img_df["column"].iloc[i]] = corr_colors[j]
            color_count[corr_color_names[j]] = color_count[corr_color_names[j]] + 1
df['color names'] = color_names
df['number of legos needed'] = color_count
needed_df = df[['color names','number of legos needed']]
new_img = Image.fromarray(img_mat, 'RGB')
new_img.show()
print(needed_df)
