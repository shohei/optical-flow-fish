import cv2
import streamlit as st
import numpy as np

st.title('Optical flow demo')
default_threshold = 141
default_pix = 4

threshold  = st.sidebar.slider("Contour: threshold", 0, 255, default_threshold)
pix = st.sidebar.slider("Contour: pixel to calculate average", 1, 20, default_pix)
f1 = st.empty()
f2 = st.empty()
f3 = st.empty()

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))

cap = cv2.VideoCapture("fish2.mp4")
_, init_frame = cap.read()
mask = np.zeros_like(init_frame)
while True:
    ret, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_color = img.copy()
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    #第一引数で指定したオブジェクトgrayscale_imgを輝度で平均化処理する。第二引数は平均化するピクセル数で、今回の場合は9,9は9x9ピクセルの計81ピクセル。
    img_blur = cv2.blur(img_gray,(pix,pix)) 
    #オブジェクトimg_blurを閾値threshold = 100で二値化しimg_binaryに代入
    ret, img_binary = cv2.threshold(img_blur, threshold, 255, cv2.THRESH_BINARY) 
    #二値化した画像オブジェクトimg_binaryに存在する輪郭を抽出
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    #抽出した輪郭の情報を用いて、オブジェクトimg_colorに書き出す
    img_color_with_contours = cv2.drawContours(img_color, contours, -1, (0,255,0), 2) 
    f1.image(img, caption='Original image',width=450)
    f2.image(img_color_with_contours, caption='Contour',width=450)

    old_frame = img.copy()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    _,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        a = int(a)
        b = int(b)
        c = int(c)
        d = int(d)
        mask = cv2.line(mask, (a,b),(c,d), (0,0,255), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img_optical = cv2.add(frame,mask)
    img_optical = cv2.cvtColor(img_optical, cv2.COLOR_BGR2RGB)
    f3.image(img_optical, caption='Optical flow',width=450)

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
