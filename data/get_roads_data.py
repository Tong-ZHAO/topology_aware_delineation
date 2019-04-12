#!/usr/bin/env python
# coding: utf-8
# Author: Tong ZHAO


import numpy 
import urllib.request 

import os, sys
import requests

list_urls = ['http://www.cs.toronto.edu/~vmnih/data/mass_roads/train/sat/index.html',
             'https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/map/index.html',
             'https://www.cs.toronto.edu/~vmnih/data/mass_roads/valid/sat/index.html',
             'https://www.cs.toronto.edu/~vmnih/data/mass_roads/valid/map/index.html',
             'https://www.cs.toronto.edu/~vmnih/data/mass_roads/test/sat/index.html',
             'https://www.cs.toronto.edu/~vmnih/data/mass_roads/test/map/index.html']

root = "massachusetts_roads_dataset/"
#list_pathes = ['train_data', 'train_pred', 'val_data', 'val_pred', 'test_data', 'test_pred']
list_pathes = ['train_pred', 'val_data', 'val_pred', 'test_data', 'test_pred']

if not os.path.exists(root):
    os.mkdir("massachusetts_roads_dataset/") 
    for path in list_pathes:
        os.mkdir(root + path)

for i, url in enumerate(list_urls):

    with urllib.request.urlopen(url) as response: 
        html = response.read().decode('utf-8')
    
    html = html.split("\n")[:-1]

    base = list_pathes[i]

    for link in html:
        adress = link[9:-10].split(">")[0][:-1]
        name = link[9:-10].split(">")[1]

        with open(os.path.join(root + base, name), "wb") as fp:
            r = requests.get(adress)  
            fp.write(r.content)