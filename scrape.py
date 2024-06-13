import requests
from bs4 import BeautifulSoup

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
file = "colors_are_in_there.html"
h_file = open(file,'r')
soup = BeautifulSoup(h_file, "html.parser")
results = soup.find("tr")
colors = results.find_all("div",class_="pciColorTabListItem")
c = []
for color in colors:
    num = color["style"].find("#")
    if num!= -1:
        c.append(color["style"][num:num+7])
        #print(color["style"][num:num+7], end="\n"*2)

rgb_vals = [hex_to_rgb(col) for col in c]
rgb_colors = []
[rgb_colors.append(x) for x in rgb_vals if x not in rgb_colors]
print(rgb_colors)
