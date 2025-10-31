import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import os

# Crear carpeta prediction si no existe
os.makedirs('prediction', exist_ok=True)

# App
def predictDigit(image):
    model = tf.keras.models.load_model("model/handwritten.h5")
    image = ImageOps.grayscale(image)
    img = image.resize((28,28))
    img = np.array(img, dtype='float32')
    img = img/255
    img = img.reshape((1,28,28,1))
    pred = model.predict(img)
    result = np.argmax(pred[0])
    return result

# Streamlit 
st.set_page_config(page_title='Reconocimiento de Dígitos ✍️', layout='wide')

# Título con estilo
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Reconocimiento de Dígitos Escritos a Mano ✍️</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #555555;'>Dibuja un dígito en el panel y presiona 'Predecir'</h4>", unsafe_allow_html=True)
st.markdown("---")

# Canvas parameters
stroke_width = st.slider('🖌️ Selecciona el ancho de línea', 1, 30, 15)
stroke_color = '#FFFFFF'  # color del trazo
bg_color = '#000000'      # fondo del canvas

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # relleno opcional
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=200,
    width=200,
    key="canvas",
)

# Mostrar la imagen dibujada
if canvas_result.image_data is not None:
    st.subheader("👁️ Vista previa de tu dibujo")
    preview_img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
    st.image(preview_img, use_column_width=False, width=150)

# Add "Predict Now" button
if st.button('🔍 Predecir'):
    if canvas_result.image_data is not None:
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'),'RGBA')
        input_image.save('prediction/img.png')
        img = Image.open("prediction/img.png")
        res = predictDigit(img)
        st.success(f"✅ El dígito es: {res}")
    else:
        st.warning('⚠️ Por favor dibuja un dígito en el canvas antes de predecir.')

# Sidebar
st.sidebar.title("ℹ️ Acerca de")
st.sidebar.write("Esta aplicación evalúa la capacidad de un RNA para reconocer dígitos escritos a mano.")
st.sidebar.write("Basado en el desarrollo de Vinay Uniyal")
st.sidebar.write("Puedes dibujar cualquier dígito del 0 al 9 en el panel principal.")
