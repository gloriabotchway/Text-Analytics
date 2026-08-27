#Import packages
import streamlit as st
import pandas as pd
import numpy as np

#Page setup
st.set_page_config(
    page_title="Machine Learning",
    page_icon="",
    layout="centered"
)

#Global environment
#import dataset
dataset="/Users/gloriabotchway/PycharmProjects/PythonProject/Labelled_stories.txt"
with open (dataset, 'r', encoding="UTF-8") as file:
    lines = file.readlines()



#local

#new*

def page1():
    st.title("Corpora Viewer")

    if st.checkbox("Raw data"):
        st.write(lines)

    if st.checkbox("Tabular representation of data"):
        st.write(lines)

    if st.checkbox("Click to upload file"):
        st.file_uploader(label="Click to upload your file", type=['csv', 'pdf', 'docx'])
def page2():
    st.title("Data Preprocessing")


def page3():
    st.title("Sentiment Analysis")

def page4():
    st.subheader("Evaluation")


#Sidebar Navigation
pages={
    'Corpora viewer': page1,
    'Data preprocessing': page2,
    'Sentiment analysis': page3,
    'Evaluation': page4
}

selected_page = st.sidebar.selectbox("Select page", list(pages.keys()))
pages[selected_page]()