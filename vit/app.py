import streamlit as st
from fastai.vision.all import load_learner, PILImage

clouds = [
    "Cirriform Clouds",
    "High Cumuliform Clouds",
    "Stratocumulus Clouds ",
    "Cumulus Clouds",
    "Cumulonimbus Clouds",
    "Stratiform Clouds",
    "Clear Sky",
]

clouds_desc = [
    "เมฆชั้นสูง ประกอบไปด้วยอนุภาคขนาดเล็ก ส่วนใหญ่ประกอบไปด้วยผลึกน้ำแข็ง เมฆมีลักษณะเป็นสีขาวและโปร่งใส",
    "เมฆที่ก่อตัวในแนวตั้ง มีอากาศปริมาณมากไหลขึ้นช่วยสนับสนุนการก่อตัวแนวในตั้ง",
    "เมฆที่มีลักษณะเป็นก้อน ลอยติดกันเป็นแพ ไม่มีรูปทรงที่ชัดเจน มีช่องว่างระหว่างก้อนเพียงเล็กน้อย มักเกิดขึ้นเวลาที่อากาศไม่ดี และมีสีเทา เนื่องจากลอยอยู่ในเงาของเมฆชั้นบน",
    "เมฆก้อนปุกปุย สีขาวเป็นรูปกะหล่ำ ก่อตัวในแนวตั้ง เกิดขึ้นจากอากาศไม่มีเสถียรภาพ ฐานเมฆเป็นสีเทาเนื่องจากมีความหนามากพอที่จะบดบังแสง จนทำให้เกิดเงา มักปรากฏให้เห็นเวลาอากาศดี ท้องฟ้าเป็นสีฟ้าเข้ม",
    "เมฆที่มีการก่อตัวในแนวตั้ง พัฒนามาจากเมฆคิวมูลัส มีขนาดใหญ่มากปกคลุมพื้นที่ครอบคลุมทั้งจังหวัดทำให้เกิดพายุฝนฟ้าคะนอง หากกระแสลมชั้นบนพัดแรง ก็จะทำให้ยอดเมฆรูปกะหล่ำกลายเป็นรูปทั่งตีเหล็ก",
    "เมฆที่มีลักษณะเป็นแผ่น แบนราบและแผ่ออกในแนวนอนต่างจาก Cumuliform และเป็นตัวบ่งชี้ว่าอากาศโดยรอบมีเสถียรภาพมากกว่ารอบๆ",
    "ท้องฟ้าแจ่มใสหรือไม่มีเมฆในบรรยากาศ",
]


def perdict(img, learn):
    pred, pred_idx, pred_prob = learn.predict(img)
    st.header(f" {clouds[int(pred)]} : {pred_prob[pred_idx]:.4f}")
    st.subheader("Description")
    st.write(f"{clouds_desc[int(pred)]}")


st.title("NephoReg")
st.write("")

learn = load_learner("new_vit_pseudo_2.pkl", cpu=False)

image_up = st.file_uploader(
    "Upload cloud image. (jpg webp) ", type=["jpeg", "jpg", "webp"]
)

if image_up is not None:
    print("Image Uploaded")
    img = PILImage.create(image_up)
    st.image(img, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    perdict(img, learn)
