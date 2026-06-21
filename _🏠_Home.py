import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

st.set_page_config(page_title="SHOPPER SPECTRUM", page_icon="🛒")

st.success("YOU ENTERNED THE APPLICATION SUCESSFULLY")

#st.image("Networking_Service_Banner3.jpg",width=700)

st.title(" 🛒🤑 SHOPPER SPECTRUM ")
st.write("ITEM RECOMENDATTION AND SHOPPING PREFRRENCES APPLCIATION")


st.sidebar.success("CHOOSE ONE OF THE OPTIONS FROM THE SIDEBAR TO GET STARTED",width=800)
