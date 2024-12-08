import whisper
import ollama


@__cached__
def load_whisper():
    return whisper.load_model("Base")

def whisper_transcribe(whisperModel, relPath: str, fp16Bool: bool) -> str: #Returns JSON Oject
    return whisperModel.transcribe(relPath, fp16 = fp16Bool)

def extract_whisper_text(transcription: str):
    return transcription['text'].strip().lower()

class Prompt():
    def __init__(self):
        self.role = ""
        self.addendums = ""
        self.extra_info = ""
        self.eval_message = ""
        self.addition_addendum = ""
        
    def set_role(self, text: str) -> str:
        self.role = text
    
    def set_addendums(self, text: str) -> str:
        self.addendums = text
        
    def set_extra_info(self, text: str) -> str:
        self.extra_info = text
        
    def set_eval_message(self, text: str) -> str:
        self.eval_message = text
        
    def set_addition_addendum(self, text: str) -> str:
        self.addition_addendum = text
        
    def generate_prompt(self, input: str) -> str:
        prompt = f'''
        {self.role}
        
        {self.addendums}
        
        Here is additional information about your task: 
        {self.extra_info}
        
        Here is the string you will evaluate: {input}
        
        {self.addition_addendum}
        '''
        
        return prompt
    
def ollama_chata(modelName: str, promptValue: str) -> str:
    return ollama.chat(
        model = modelName,
        messages = [{'role':'user', 'content':promptValue}]
    )