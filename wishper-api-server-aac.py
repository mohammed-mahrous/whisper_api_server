from functools import reduce
from flask import Flask, jsonify , request
from whisper_online import *
from faster_whisper import *
import os , datetime
from pydub import AudioSegment
from typing import IO , Any
from os import PathLike


RATE:int = 16000
HOST:str = '10.143.232.21'
PORT:int = 5000

src_lan:str = "ar"
tgt_lan:str = "ar"
model_size:str = 'large-v3'
test_model_size:str = 'small'
device:str = 'cuda'
test_device:str ='cpu'
test_compute_type = 'int8'
model:FasterWhisperASR = FasterWhisperASR(lan=tgt_lan, modelsize=model_size,device=device)


def exportFile(file) -> IO[Any] | Any | PathLike: 
    seg:AudioSegment = AudioSegment.from_file(file.stream, 'mp3')
    exported = seg.export('temp-file-{}.mp3'.format(getFormattedDate()))

    return exported


def handleSegments(segments:list) -> str:
    o = []
    for segment in segments:
        o.append(segment.text)
    text = " \r\n\r\n ".join([txt for txt in o])
    return text


def cleanFilesCache() -> None:
    pass

def wavToText(file_dest:str, translate:bool=False) -> str:
    try:
        if(translate):model.set_translate_task()
        else: model.transcribe_kargs.clear()
        segments = model.transcribe(file_dest);
        if(segments):
            return handleSegments(segments);
        return ''
    except Exception as e:
        print(f"Some Error! {e} happened while generating text for {file_dest}")
        
        
def clean_text(text: str, en: bool = False) -> str:
    if not text:
        return ''
    phrases = {
        'en': [
            'Welcome to my channel \r\n\r\n Thank you for watching',
            'Welcome to my channel',
            'Thank you for watching',
            'The car is parked in the middle of the road.',
        ],
        'ar': ['اشتركوا في القناة']
    }
    
    target_phrases = phrases['en' if en else 'ar']
    
    if '\r\n\r\n' in text:
        for phrase in target_phrases:
            text = text.replace(phrase,'')
        if text == "  \r\n\r\n  ": return ''
        else : return text
    return '' if any(phrase in text for phrase in target_phrases) else text


app = Flask(__name__)

def getFormattedDate() -> str:
    now = datetime.datetime.now()
    return f'{now.day}-{now.hour}-{now.minute}-{now.second}'

@app.route('/transcript/aac', methods = ['POST'])
def transcript():
    try:
        print(request.form)
        
        if('file' in request.files):
            translate:bool = request.form['translate'].lower() == "yes"
            file = request.files['file']
            exported = exportFile(file=file)
            text_result = wavToText(exported.name)
            text_result = clean_text(text_result)
            
            if(translate):
                translation = wavToText(exported.name,translate=True)
                translation = clean_text(translation,en=True)
            exported.close()
            os.remove(exported.name)
            print(f'transcript: {text_result}') 
            print(f'translation: {translation}')
            return jsonify({"data":{'transcript': text_result, 'translation': translation if(translation) else ''}}) if(translate) else jsonify({'data': {'transcript':text_result}})
        jsonify({"error": "No file provided"}), 400
    except Exception as e:
        print(e)
        return jsonify({f"e: {e}"}), 500

if __name__ == '__main__':
    # from waitress import serve
    # serve(app, host=HOST, port=PORT)
    app.run(host=HOST,debug=False, port=PORT)
