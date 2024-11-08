import speech_recognition as sr
import pandas as pd
import re

# Initialize speech recognizer
recognizer = sr.Recognizer()

def transcribe_audio():
    with sr.Microphone() as source:
        print("Please dictate your notes:")
        audio = recognizer.listen(source)
        transcript = recognizer.recognize_google(audio)
        return transcript

def parse_transcript(transcript):
    # Example parsing with regex
    patient_name = re.search(r'Patient Name: (.+)', transcript)
    age = re.search(r'Age: (\d+)', transcript)
    symptoms = re.search(r'Symptoms: (.+)', transcript)
    diagnosis = re.search(r'Diagnosis: (.+)', transcript)

    # Create a dictionary to hold the extracted data
    data = {
        'Patient Name': patient_name.group(1) if patient_name else None,
        'Age': age.group(1) if age else None,
        'Symptoms': symptoms.group(1) if symptoms else None,
        'Diagnosis': diagnosis.group(1) if diagnosis else None
    }
    return data

# Transcribe dictation and parse text
transcript = transcribe_audio()
parsed_data = parse_transcript(transcript)

# Convert parsed data to a DataFrame and save to Excel
df = pd.DataFrame([parsed_data])
df.to_excel('patient_data.xlsx', index=False)

print("Data has been saved to patient_data.xlsx")
