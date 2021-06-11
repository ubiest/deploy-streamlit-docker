# frontend/main.py

import requests
import streamlit as st
from PIL import Image
import altair as alt
import time
from fastai.vision.all import *


file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg","webp" ])
class_btn = st.button("Classify")
if file_uploaded is not None:
    #image = PILImage.create(file_uploaded)
    image = Image.open(file_uploaded)
    st.image(image, caption='Uploaded Image', use_column_width=True)

if class_btn:
    if file_uploaded is None:
        st.write("Invalid command, please upload an image")
    else:
        with st.spinner('Model working....'):
            #### TODO passar imatge mitjançant request
            #files = {"file": file_uploaded.getvalue()}
            files = {"file": image}
            res = requests.post(f"http://backend:8080/classify", files=files)

            #predictions, pred_dict = predict(image) ### Old style direct call
            # payload = res.json()
            # predictions = payload.get('phrase')
            # pred_dict = payload.get('values')
            ###############
            try:
                payload = res.json()
                predictions = payload.get('phrase')
                pred_dict = payload.get('values')
                # img_path = res.json()
                # image = Image.open(img_path.get("name"))
                # st.image(image, width=500)

            except ValueError:
                st.write("The request code is: " + str(res.status_code))
                st.write("The request content is: " + str(res.text))
            #image = Image.open(img_path.get("name"))


            try:
                time.sleep(1)
                st.success('Classified')
                st.write(predictions)
            except NameError:
                st.write('No image received, nor classified')

            if pred_dict is not None:
                df = pd.DataFrame.from_dict(pred_dict, orient='index').reset_index()
                df.columns = ['Room', 'Score']
                ### Generate a bar graph with scores
                bars = alt.Chart(df).mark_bar().encode(
                    y = 'Room',
                    x = 'Score:Q'
                )

                text = bars.mark_text(
                    align='left',
                    baseline='middle',
                    dx=3  # Nudges text to right so it doesn't appear on top of the bar
                ).encode(
                    text='Score:Q'
                )

                (bars + text).properties(height=900)
                st.altair_chart((bars + text))
