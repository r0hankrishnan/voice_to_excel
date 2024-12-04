import streamlit as st
import numpy as np
import ollama
import whisper
import pandas as pd

if "data" not in st.session_state:
    st.session_state["data"] = []

@st.cache_resource
def generate_whisper():
    model = whisper.load_model("base")
    return model

model = generate_whisper()

st.header("transcribe notes")

audio_data = st.audio_input(label="record note here")

if audio_data is not None:
    # Save the recorded audio for processing
    audio_path = "recorded_audio.wav"
    with open(audio_path, "wb") as f:
        f.write(audio_data.getvalue())
        
    st.write("Transcribing audio...")
    result = model.transcribe(audio_path, fp16=False)
    transcribed_text = result['text']
    st.write("Transcription complete:")
    st.text(transcribed_text)
    
    st.write("Formatting transcription...")
    prompt = f'''
    You will receive a string. Your job is to parse the text and create a python dict that maps the category (e.g., patient name) to its value.
    DO NOT RETURN ANY PREAMBLE. ONLY RETURN THE PYTHON DICTIONARY WITH THE PARSED FIELDS. 
    I DO NOT WANT A PYTHON FUNCTION. DO NOT INCLUDE ANY PUNCTUATION IN THE VALUES OF THE DICTIONARY.
    I WANT YOU TO ACTUALLY CREATE AND RETURN THE DICTIONARY.
    
    Here is an example:
        "dict(input: "patient name, kevin bacon, date of service, may 23rd, 2024, level of service, annual visit, 
                location, hospice, code 1, heart attack, code 2, tiny balls.",
        output: dict('patient_name:': 'Kevin Bacon', 'date_of_service': '05-23-2024', 
                'level_of_service': 'annual visit', 'location':'hospice', 'code_1': 'heart attack', 'code_2': 'tiny balls')") 

    Here is the string you will evaluate: {transcribed_text.strip().lower()}. 
    '''
    response = ollama.chat(
        model="llama3.2",
        messages=[{'role': 'user', 'content': prompt}]
    )
    formatted_dict = eval(response['message']['content'])
    st.write("Formatting complete:")
    st.json(formatted_dict)

    st.write("Converting to DataFrame...")
    df = pd.DataFrame([formatted_dict])
    st.dataframe(df)
    
    st.session_state["data"].append(formatted_dict)



# Convert to DataFrame and Display
if st.session_state["data"]:
    st.write("Current Dataset:")
    df = pd.DataFrame(st.session_state["data"])
    st.dataframe(df)
    
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')


    csv = convert_df(df)

    st.download_button(
    "Press to Download",
    csv,
    "notes.csv",
    "text/csv"
    )

# Reset Button
if st.button("Reset"):
    st.session_state.clear()  # Clear the session state
    st.session_state["data"] = [] 
    
