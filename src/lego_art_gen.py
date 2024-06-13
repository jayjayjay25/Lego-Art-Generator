import pandas as pd
from PIL import Image
import sklearn as sk

def make_pic(filename):
    img = Image.open(filename)
    #img = Image.open('lego_test_monroe.png')
    img = img.convert('RGB')
    newsize = (53,53)
    img = img.resize(newsize)
    img.show()
    img_mat = np.array(img)
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
    return (img,image_df)



df = pd.DataFrame(columns = ["R","G","B"])
#filename = 'The_Garden_of_earthly_delights.jpeg'
filename = 'lego_test_monroe.png'
#filename = 'melvin.jpeg'


