from PIL import Image
import streamlit as st
from fastai.vision.all import load_learner, PILImage

def perdict(img,learn):
    pred,pred_idx,pred_prob = learn.predict(img)
    st.header(f' class : {pred} prob : {pred_prob[pred_idx]}')

st.title("NephoReg")
st.write("")

learn = load_learner('new_vit.pkl', cpu=False)

image_up = st.file_uploader("Upload cloud image. (jpg webp) ", type=["jpeg","jpg","webp"])

if image_up is not None:
    print("Image Uploaded")
    img = PILImage.create(image_up)
    st.image(img, caption = 'Uploaded Image.', use_column_width = True)
    st.write("")
    perdict(img,learn)
