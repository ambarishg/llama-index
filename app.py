import streamlit as st

from  llam_index_helper import *

st.title("Question Answering System of PWD")

prompt = st.text_input('Your Question')

if st.button('Submit'):
    st.write(str(get_response(prompt)))