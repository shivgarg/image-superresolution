from PIL import Image
import os
import argparse
from multiprocessing import Pool

args = argparse.ArgumentParser()
args.add_argument("src")
args.add_argument("dest")

size = 256, 256

args = args.parse_args()
os.makedirs(args.dest, exist_ok=True) 


images = []
for f,_,imgs in os.walk(args.src):
    images += map(lambda a: (f,a), imgs)
        



def resize(tup):
        f = tup[0]
        img = tup[1]
        im = Image.open('/'.join([f,img]))
        im_resized = im.resize(size)
        im_resized.save('/'.join([args.dest,os.path.relpath(f,args.src),img]))

p = Pool(4)
p.map(resize,images)
