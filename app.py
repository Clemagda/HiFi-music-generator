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


def generate_music_tensor(description, duration: int):
    """Generate aduio tensors and tokens 

    Args:
        description (_type_): string
        duration (int): _description_

    Returns:
        torch tensors: first dimension of tensor generated
    """
    print(f"Description : {description}")
    print(f"Duration: {duration}")
    model = load_model()

    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        duration=duration
    )

    outputs = model.generate(descriptions=description,
                             progress=True,
                             return_tokens=True)

    return outputs[0]


def save_audio(samples: torch.tensor):
    """Save generated audio to .wav format

    Args:
        samples (torch.tensor): _description_
    """
    sample_rate = 32000,
    save_path = "audio_outputs/"

    assert samples.dim() == 2 or samples.dims() == 3

    samples = samples.detach().cpu()

    if samples.dim() == 2:
        samples = samples[None, ...]

        for idx, audio in enumerate(samples):
            audio_path = os.path.join(save_path, f"audio_{idx}.wav")
            torchaudio.save(audio_path, audio, sample_rate)


def get_binary_file_downloader_html(bin_file, file_label='File'):
    """Download generated audio file.

    Args:
        bin_file (_type_): _description_
        file_label (str, optional): _description_. Defaults to 'File'.

    Returns:
        _type_: _description_
    """
    with open(bin_file, 'rb') as f:
        data = f.read()

    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream; base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}<a>'
    return href


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
    time_slider = st.slider("Select time duration (in seconds)", 2, 20, 5)

    if text_area and time_slider:
        st.json(
            {
                "Your description": text_area,
                "Selected Time Duration": time_slider
            }
        )
        st.subheader("Generating...")

        music_tensors = generate_music_tensor(text_area, time_slider)

        print(f"Music tensors : {music_tensors}")

        save_music_file = save_audio(music_tensors)
        audio_file_path = "audio_outputs/audio_0.wav"
        # TODO: modifier audio_file_path pour une indexatio dynamique.Actullement, aucun audio n'est gardé en mémoire.
        audio_file = open(audio_file_path, 'rb')
        audio_bytes = audio_file.read()

        st.audio(audio_bytes)
        st.markdown(get_binary_file_downloader_html(
            audio_file_path, 'Audio'), unsafe_allow_html=True)


# TODO: afficher interface streamlit
if __name__ == "__main__":
    main()
