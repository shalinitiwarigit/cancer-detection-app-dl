import numpy as np
import streamlit as st

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model
model = load_model('cancer_model.h5')

# st.title("Cancer Detection App 🧬")
# st.write("I am your deep learning doctor here to help you. 🩺")
# st.markdown("🧠 This project"
#          " focuses on detecting cancer from medical images using Convolutional Neural Networks "
#          )

st.markdown("<h1 style='text-align: center; color: #6a1b9a;'>🧬 Cancer Detection App</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Your AI-powered Deep Learning Doctor</h3>", unsafe_allow_html=True)



upload=st.file_uploader("Upload the Image")

if upload is not None:
    img=Image.open(upload).convert('RGB')

    img=img.resize((150,150))
    img_array=image.img_to_array(img)
    img_array=np.expand_dims(img_array,axis=0)
    img_array/=255

    prediction=model.predict(img_array)

    if prediction[0][0]<0.5:
        # st.error("Cancer")
        st.error(
            "⚠️ **Cancer Detected**\n\nThe uploaded image shows signs that may be associated with cancer. Please consult a medical professional for a detailed diagnosis and treatment plan.")


    else:
        st.success(
            "✅ **No Cancer Detected**\n\nThe uploaded image does not show visible signs of cancer. However, always consult a medical expert for accurate confirmation.")


