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


st.header("Transcribe your notes")

with st.expander(label="Instructions"):
    st.markdown('''
                **Please make sure to read your notes in the order of the table below:**
                
                You can read your notes as "Data Name, Value, Data Name, Value, Data Name, Value, etc."
                
                The decoder model is somewhat sensitive to naming discrepancies so please just use the column names or it might not parse your notes properly!
                ''')
    st.image("./assets/data_fields.png")



st.subheader("Here are the fields in order:")

col1, col2 = st.columns(2)
with col1:
    st.markdown('''
                1. Patient Name:
                2. Date of Service:
                3. Level of Service:
                4. Location: 
                5. Code 1:
                6. Code 2:
                7. Code 3:
                8. Code 4:
                ''')

with col2:
    st.markdown('''                
                9. Code 5:
                10. Code 6:
                11. Code 7:
                12. Code 8:
                13. Code 9:
                14. Code 10:
                15. Code 11:
                16. Code 12:
                ''')

audio_data = st.audio_input(label="Record note here")

if audio_data is not None:
    with st.spinner("Transcribing audio..."):
        # Save the recorded audio for processing
        audio_path = "recorded_audio.wav" #Set path
        with open(audio_path, "wb") as f: #Open wav file and write bytes
            f.write(audio_data.getvalue())
            
        # st.write("Transcribing audio...") #Processing message
        
        with st.container(border = True): #Display transcription in a box
            result = model.transcribe(audio_path, fp16=False) #Transcription
            transcribed_text = result['text']
            st.write("Transcription complete:")
            st.text(transcribed_text)
    
    with st.spinner("Formatting transcription..."):
        # st.write("Formatting transcription...")
        prompt = f'''
        You will receive a string. Your job is to parse the text and create a python dict that maps the category (e.g., patient name) to its value.
        DO NOT RETURN ANY PREAMBLE. ONLY RETURN THE PYTHON DICTIONARY WITH THE PARSED FIELDS. 
        I DO NOT WANT A PYTHON FUNCTION. DO NOT INCLUDE ANY PUNCTUATION IN THE VALUES OF THE DICTIONARY.
        I WANT YOU TO ACTUALLY CREATE AND RETURN THE DICTIONARY.
        
        The dictionary will always have the following keys (in this order):
        
        patient_name, date_of_service, level_of_service, location, code_1, code_2, code_3, code_4, code_5, code_6, 
        code_7, code_8, code_9, code_10, code_11, code_12
        
        Output a dictionary with these keys in the above order. It is your job to map the information in the transcribed text into its proper category. If there are less than 12 codes, 
        fill in as many code columns as needed and leave the other code columns empty. Always output the date of service in the format of "MM-DD-YYYY".

        Here is the string you will evaluate: {transcribed_text.strip().lower()}. 
        '''
        response = ollama.chat(
            model="llama3.2",
            messages=[{'role': 'user', 'content': prompt}]
        )
        formatted_dict = eval(response['message']['content'])
        with st.container(border=True):
            st.write("Formatting complete:")
            st.json(formatted_dict)

        with st.spinner("Converting to DataFrame..."):
            # st.write("Converting to DataFrame...")
            df = pd.DataFrame([formatted_dict])
        with st.container(border=True):
            st.write("New note:")
            st.dataframe(df)
        
        st.session_state["data"].append(formatted_dict)



# Convert to DataFrame and Display
if st.session_state["data"]:
    with st.container(border=True):
        st.write("Current Dataset:")
        fdf = pd.DataFrame(st.session_state["data"])
        st.dataframe(fdf)
    
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')


    csv = convert_df(fdf)

    st.download_button(
    "Press to Download",
    csv,
    "notes.csv",
    "text/csv"
    )

    st.markdown('''
            After exporting your csv file, you can do the following to clear your data so that the application is fresh for the next day you need to compile notes:
            1. Click the "trash" icon in the upper right section of the audio input widget.
            2. Wait until all model outputs clear.
            3. **Double click** the reset button.
            ''')
    
# Reset Button
if st.button("Reset"):
    st.session_state.clear()  # Clear the session state
    st.session_state["data"] = [] 
    
