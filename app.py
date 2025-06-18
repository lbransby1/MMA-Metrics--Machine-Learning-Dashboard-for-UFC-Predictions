import streamlit as st

def greet(name):
    return "Hello " + name + "!! :O"

st.title("Greeting App")

name = st.text_input("Enter your name")

if st.button("Greet"):
    greeting = greet(name)
    st.write(greeting)
