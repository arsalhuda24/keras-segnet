from skimage.transform import rescale, resize, downscale_local_mean
from skimage.io import imread,imsave
import pandas as pd

path1='Data/'
# df= pd.read_csv("/Users/Arsal/Downloads/keras-segnet-master/Data/train1.csv")
df = pd.read_csv(path1 + 'train1' + '.csv')
s= imread("/Users/Arsal/Downloads/keras-segnet-master/Data/train/JPCLN001.jpg")
print(s.shape)
a=w
# print(df)
for i,(img,gt) in df.iterrows():
   print(i)
   image=imread(path1+gt)
   image=resize(image,(256,256))
   imsave('Label'+str(i)+'.jpg',image)