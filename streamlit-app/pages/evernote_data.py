import streamlit as st
import pandas as pd

df = pd.read_csv("./data-assets/enex-parsed.csv", index_col=0)

for row in df.iterrows():
    note = row[1]
    with st.expander(f"{note.title}"):
        st.write(note.content)