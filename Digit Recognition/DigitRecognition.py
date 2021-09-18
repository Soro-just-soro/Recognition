import cv2
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl,time
if(not os.environ.get('PYTHONHTTPSVERIFY','')and getattr(ssl,'_create_unverifed_context',None)):
    ssl._create_default_https_context=ssl._create_unverifed_contents
x,y=fetch_openml('mnist_784',version=1,return_X_y=True)
print(pd.Series(y).value_counts())
classes=['0','1','2','3','4','5','6','7','8','9']
nclasses=len(classes)
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=9, train_size=7500, test_size=2500)
#scaling the features
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
cap=cv2.VideoCapture(0)
while(True):
    try:
        ret,frame=cap.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height,width=gray.shape
        upperleft=(int(width/2-56),int(height/2-56))
        bottomright=(int(width/2+56),int(height/2+56))
        cv2.rectangle(gray,upperleft,bottomright,(0,255,0),2)
        roi=gray[upperleft[1]:bottomright[1],upperleft[0]:bottomright[0]]
        impil=Image.fromarray(roi)
        ImageBW=impil.convert('L')
        IBWR=ImageBW.resize((28,28),Image.ANTIALIAS)
        ImageInverted=PIL.ImageOps.invert(IBWR)
        PixelFilter=20
        minPixel=np.percentile(ImageInverted,PixelFilter)
        Imagescaled=np.clip(ImageInverted-minPixel,0,255)
        maxPixel=np.max(ImageInverted)
        Imagescaled=np.asarray(Imagescaled)/maxPixel
        testSample=np.array(Imagescaled).reshape(1,784)
        testpredict=clf.predict(testSample)
        print("predicteddigitis",testpredict)
        cv2.imshow('frame',gray)
        if cv2.waitKey(1)& 0xFF==ord('q'):
            break
    except Exception as e:
        pass

cap.release()
cv2.destroyAllWindows()
