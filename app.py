import cv2
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tempfile

st.title('Optical flow demo')
default_threshold = 141
default_pix = 4

threshold  = st.sidebar.slider("Contour: threshold", 0, 255, default_threshold)
pix = st.sidebar.slider("Contour: pixel to calculate average", 1, 20, default_pix)
f1 = st.empty()
f2 = st.empty()
f3 = st.empty()
f4 = st.empty()
f5 = st.empty()

is_contour_enabled = st.sidebar.toggle('Contour')
is_optical_flow_enabled = st.sidebar.toggle('Optical flow')

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

tmpfile_name = tempfile.NamedTemporaryFile().name+'fig1.png'
print(tmpfile_name)

cap = cv2.VideoCapture("fish2.mp4")
_, init_frame = cap.read()
hsv = np.zeros_like(init_frame)
hsv[...,1] = 255
average_flow_array = []

while True:
    ret, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    f1.image(img, caption='Original image',width=450)

    img_color = img.copy()
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    if is_contour_enabled:
        #第一引数で指定したオブジェクトgrayscale_imgを輝度で平均化処理する。第二引数は平均化するピクセル数で、今回の場合は9,9は9x9ピクセルの計81ピクセル。
        img_blur = cv2.blur(img_gray,(pix,pix)) 
        #オブジェクトimg_blurを閾値threshold = 100で二値化しimg_binaryに代入
        ret, img_binary = cv2.threshold(img_blur, threshold, 255, cv2.THRESH_BINARY) 
        #二値化した画像オブジェクトimg_binaryに存在する輪郭を抽出
        contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
        #抽出した輪郭の情報を用いて、オブジェクトimg_colorに書き出す
        img_color_with_contours = cv2.drawContours(img_color, contours, -1, (0,255,0), 2) 
        f2.image(img_color_with_contours, caption='Contour',width=450)

    if is_optical_flow_enabled:
        prvs = img_gray.copy()
        ret, frame2 = cap.read()
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        f3.image(bgr, caption='Optical flow',width=450)
        average_flow = np.mean(mag)
        f4.text('Average flow: ' + str(average_flow) + ' pixel/frame')

        average_flow_array.append(average_flow)
        fig, ax = plt.subplots(figsize=(5,2.5))
        plt.title('Average flow')
        plt.xlabel('Frame')
        plt.ylabel('Average flow (pixel/frame)')
        ax.plot(average_flow_array)

        fig.savefig(tmpfile_name)
        fig_image = Image.open(tmpfile_name)
        f5.image(fig_image)

        prvs = next
