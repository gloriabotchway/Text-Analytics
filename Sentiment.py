#Panda
#numpy
#streamlit or Django Flask ....... libraries for creating a website or a web project
#plotly or matplotlib
#Scikit-learn

import streamlit as st
number1 = 47
number2 = 78
Total = number1 + number2
st.write(Total)

st.header("file upload")
st.subheader("file upload")
st.caption("file upload")
st.write("file upload")

number1 =st.number_input("Enter a number 1")
number2 =st.number_input("Enter a number 2")
number3 =st.number_input("Enter a number 3")
st.button('calculate')

file = st.file_uploader("Click to Upload file or drag to drop file", type=['pdf'])