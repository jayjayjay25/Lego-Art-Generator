import warnings
import pandas as pd
import numpy as np
import itertools as itz
import plotly.express as px
import plotly.graph_objects as pxg
import statistics as st
import random as ran
from PIL import Image
import math as m
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
# Import ElbowVisualizer
import sklearn as sk
from sklearn.cluster import KMeans
import yellowbrick as yb
from yellowbrick.cluster import KElbowVisualizer
##from IPython.display import display
#import seaborn as sn

def rounding(floater: int):
    if abs(floater-m.ceil(floater)) > abs(floater-m.floor(floater)):
        new_num = m.floor(floater)
    else:
        new_num = m.ceil(floater)
    return int(new_num)
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
def kmeans(img_df: pd.DataFrame, cluster_count: int, list_of_clusters: list):#,list_of_indices):#list of clusters should be a list of dim 3 tuples (or "VOID")
    #img_df: the dataframe with the RBG values (listed column-wise) of each pixel in the image (listed row-wise)
    #cluster_count: value of k
    #list_of_clusters: a list of 3-tuples, each 3-tuple is a centroid chosen from the last kmeans iteration. If its the first iteration, however, list of clusters has
    #only the element "VOID" inside it. 
    
    #setting up the initial cluster centroids
    dee_ef = img_df.transpose() #transposing it makes it a little easier to work with I hate to say...
    #print("Dee_ef: ")
    #print(dee_ef)
    if list_of_clusters[0]=="VOID": #initial random sampling of values from dee_ef to serve as centroids
        list_of_names = list(dee_ef.columns.values)
        init_centroids = ran.sample(list_of_names, cluster_count)
        ef_dee = dee_ef[init_centroids].transpose()
        df_centroids = np.array(ef_dee)
    else:
        df_centroids = list_of_clusters

    #print("Initial Centroids: ")
    #print(df_centroids)
    clustermat = np.zeros((len(img_df),cluster_count))
    distlist = []
    mindex =0
    #kmeans does its thing
    for c in dee_ef:
        for k in range(0,len(df_centroids)):
            pixelQ = dee_ef[c] #the 3-tuple values of pixel c
            pixelC = df_centroids[k] #the 3-tuple values of centroid k
            dist = (pixelQ[0] - pixelC[0])**2 + (pixelQ[1] - pixelC[1])**2 + (pixelQ[2] - pixelC[2])**2
            distlist.append(dist)
        for i in range(0,len(distlist)): #find index of the centroid with minimum distance to pixel c
            if min(distlist) == distlist[i]:
                mindex = i
            clustermat[c][mindex] = 1
        distlist = []

    #getting new centroids...
    #print("Cluster matrix found: ")
    #print(clustermat)
    sum = (0,0,0)
    avgs = []
    clusterf_ck= clustermat.sum(axis=0)
    print("Cluster f_ck: ")
    print(clusterf_ck)
    for i in range(0,cluster_count):
        sum = [0,0,0]
        for j in range(0,len(img_df)):
            if clustermat[j][i]==1:
                sum[0]=sum[0] + img_df["R"].iloc[j]
                sum[1]= sum[1] + img_df["G"].iloc[j]
                sum[2]=sum[2] + img_df["B"].iloc[j]
        if clusterf_ck[i]!=0:
            length_cluster = sum/clusterf_ck[i]
            avgs.append(length_cluster)
        else:
            avgs.append(0)
    avgs = np.array(avgs)
    #print("New centroids found: ")
    #print(avgs)
    return(cluster_count,avgs,clustermat)#,indices)
#extracting color possibilities
def mat_equals(mat1,mat2,eye,jay):
    for i in range(0,eye):
        for j in range(0,jay):
            if mat1[i][j]!=mat2[i][j]:
                return 0
    return 1

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
def img_preprocessing():
    #creating the set of lego colors df
    data = make_colors()
    colors_df = pd.DataFrame(data, columns = ['color names',"R","G","B"])
    print("Colors dataframe: ")
    print(colors_df)
    color_names = colors_df['color names']
    colors_df = colors_df.drop(columns = ['color names'])
    colors_df = colors_df.astype(float)
    #new_df = colors_df.to_numpy()
    #opening the image
    #img = Image.open('The_Garden_of_earthly_delights.jpeg')
    #img = Image.open('melvin.jpeg')
    img = Image.open('starry_night.jpeg')
    #img = Image.open('the_scream.jpeg')
    #img = Image.open('time_dali.jpeg')
    #img = Image.open('ditto.jpeg')
    #img = Image.open('lego_test_monroe.png')
    #img = Image.open("rolling_stones.jpeg")
    img = img.convert('RGB')
    newsize = (53,53)
    img = img.resize(newsize)
    img.show()
    img_mat = np.array(img)
    #Flattening the image into a dataframe and recording important info about it
    img_df = np.zeros((len(img_mat)**2,5))
    img_df = pd.DataFrame(img_df,columns= ["R","G","B","row","column"])
    count = 0
    for i in range(0,len(img_mat)):
        for j in range(0,len(img_mat)):
            img_df["R"][count] = img_mat[i][j][0]
            img_df["G"][count] = img_mat[i][j][1]
            img_df["B"][count] = img_mat[i][j][2]
            img_df["row"][count] = i
            img_df["column"][count] = j
            count = count + 1
    print("Image dataframe: ")
    print(img_df)
    image_df = img_df.drop(columns = ["row","column"])
    #only needed for initial run so that I can look at the elbow point, choose a k, and then run the whole thing over again lolol
    #model = KMeans()
    #k is range of number of clusters.
    #visualizer = KElbowVisualizer(model, k=(4,70), timings=False)
    #visualizer.fit(image_df)        # Fit the data to the visualizer
    #visualizer.show()        # Finalize and render the figure
    return (img_df, img_mat, colors_df, color_names, image_df)

#main
(img_df, img_mat, colors_df, color_names, image_df) = img_preprocessing()
#image_df is the same as img_df but without considering the indexes of each pixel
list_of_clusters = []
list_of_clusters.append("VOID") #starts off with random initializations
cluster_count = 15
#repeating kmeans until convergence
#for i in range(0,50):
(cluster_count,avgs,clustermat) = kmeans(image_df,cluster_count,list_of_clusters) #initial kmeans
j=0
z = 0
while (j != 1) and (z < 400):
    count = 0
    (new_cc,new_avgs,new_cmat) = kmeans(image_df,cluster_count,avgs)
    #for i in range(0,24):
    #    if new_avgs[i] == avgs[i]:
    #        count = count + 1
    #if (new_cmat==clustermat).all():
    #for n in range(0,len(new_avgs)):
    #    if (abs(rounding(avgs[n][0]) - rounding(new_avgs[n][0]))==0) & (abs(rounding(avgs[n][1]) - rounding(new_avgs[n][1]))==0) & (abs(rounding(avgs[n][2]) - rounding(new_avgs[n][2]))==0):
    #        count = count + 1
    #if count == len(new_avgs):
    if mat_equals(new_cmat,clustermat,len(image_df),cluster_count)==1:
        print("done!")
        j = 1
    else:
        clustermat = new_cmat
        avgs = new_avgs
    z = z+1
    print(z)
print("Centroids found: ")
print(avgs)
#def img_postprocessing()
#applying data found from kmeans to choose best colors from lego's color arsenal
mindist = 1000000000000000000
mincol = [33,33,33]
mindex = 0
corr_colors = [] #stores the best lego color for cluster i
corr_color_names = [] #stores the index of the name of the best lego color for cluster i
color_count = np.zeros([len(colors_df)]) #keeps track of the amount of legos of a certain color needed
alr_chosen = []
#comparing centroids of each cluster with lego colors and finding the best fitting lego colors for each centroid
for i in range(0,len(new_avgs)):
    for j in range(0,len(colors_df)):
        dist = (colors_df["R"].iloc[j] - new_avgs[i][0])**2 + (colors_df["G"].iloc[j] - new_avgs[i][1])**2 + (colors_df["B"].iloc[j] - new_avgs[i][2])**2
        if (dist < mindist):# & (j not in alr_chosen):
            mindist = dist
            mincol = colors_df.iloc[j]
            mindex = j
            #color_names[j]
    corr_colors.append(mincol)
    corr_color_names.append(mindex)
    alr_chosen.append(mindex)
    mincol = [33,33,33]
    mindex = 0
    mindist = 1000000000000000000
print("list of indices: ")
print(alr_chosen)
#replacing each pixel's color in the image with its reassigned lego color
for j in range(0,cluster_count):
    for i in range (0,len(img_df)):   
        if new_cmat[i][j] == 1:
            #print(img_df["row"].iloc[i])
            img_mat[int(img_df["row"].iloc[i])][int(img_df["column"].iloc[i])] = corr_colors[j]
            color_count[corr_color_names[j]] = color_count[corr_color_names[j]] + 1
colors_df['color names'] = color_names
colors_df['number of legos needed'] = color_count
needed_df = colors_df[['color names','number of legos needed']]
new_img = Image.fromarray(img_mat, 'RGB')
new_img.show()
print(needed_df)
