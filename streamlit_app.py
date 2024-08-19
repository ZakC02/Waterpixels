import streamlit as st
from watershed5 import main
import matplotlib.pyplot as plt

st.title('Waterpixel App')
st.write('#### by ZakC02')
st.write('Get a superpixel segmentation of your image using the Watershed algorithm')
st.write('More details can be found [here.](https://github.com/ZakC02/Waterpixels.git)')



img = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if img is not None:
    st.image(img, caption='Original Image')
    k = st.slider('Constant k', 0, 10, 1)
    sigma = st.slider('Sigma', 1, 50, 20)
    thickness = 2
    #thickness = st.slider('Space between regular areas', 0, 10, 2)

    color = [0,0,0]
    h = st.color_picker('Chose a color for the Waterpixel boundaries', '#000000')
    h = h.lstrip('#')
    color = list(int(h[i:i+2], 16) for i in (0, 2, 4))
    output = main(img, k=k, sigma=sigma, thickness=thickness, color=color)
    st.image(output, caption="Waterpixels")
