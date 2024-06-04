# Installation des packages et bilbiothèques nécessaires

from audiocraft.models import MusicGen
import streamlit as st
import os
import torch
import torchaudio
import numpy as np
import base64


@st.cache_ressource
def load_model():
    model = MusicGen.get_pretrained('facebook/musicgen-small')
    return model


st.set_page_config(
    page_icon=":notes:",  # icones disponibles sur la doc streamlit
    page_title="Music Gen"
)


def main():
    st.title("Text to Music Generation")
    with st.expander("See Explanation"):
        st.write(
            "This is a music generation application based on Music Gen model. Based on your natural language description, it'll generate your desired music")
    text_area = st.text_area("Enter your music description here...")
    time_slider = st.slider("Select time duration (in seconds)", 2, 5, 20)

    if text_area and time_slider:
        st.json(
            {
                "Your description": text_area,
                "Selected Time Duration": time_slider
            }
        )
        st.subheader("Generating...")
    pass


# TODO: afficher interface streamlit
if __name__ == "__main__":
    main()
