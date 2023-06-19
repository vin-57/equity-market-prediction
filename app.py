import numpy as np
import streamlit as st
import sa
PAGES = {
    "Sentimental Analysis & stock market prediction": sa
}

st.sidebar.title('Finanical')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()
