from flask import Flask, jsonify , request
from whisper_online import *
from faster_whisper import *
import os , datetime
from pydub import AudioSegment
from typing import IO , Any
from os import PathLike


RATE:int = 16000
HOST:str = '127.0.0.1'
PORT:int = 5000

src_lan:str = "ar"
tgt_lan:str = "ar"
model_size:str = 'large-v3'
test_model_size:str = 'small'
device:str = 'cuda'
test_device:str ='cpu'
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

def wavToText(file_dest:str) -> str:
    try:
        segments = model.transcribe(file_dest);
        if(segments):
            return handleSegments(segments);
        return '';
    except Exception as e:
        print(f"Some Error! {e} happened while generating text for {file_dest}")
        
        
def clean_text(text: str) -> str:
    if 'اشتركوا في القناة' in text:
        return text.replace('اشتركوا في القناة', '') if '\r\n\r\n' in text else ''
    return text

app = Flask(__name__)

def getFormattedDate() -> str:
    now = datetime.datetime.now()
    return f'{now.day}-{now.hour}-{now.minute}-{now.second}'

@app.route('/transcript/aac', methods = ['POST'])
def transcript():
    try:
        if('file' in request.files):
            file = request.files['file']
            exported = exportFile(file=file)
            text_result = wavToText(exported.name)
            text_result = clean_text(text_result)
            exported.close()
            os.remove(exported.name)
            print(f'result: {text_result}') 
            return jsonify({'transcript': text_result})
        jsonify({"error": "No file provided"}), 400
    except Exception as e:
        return jsonify({f"e: {e}"}), 500

if __name__ == '__main__':
    # from waitress import serve
    # serve(app, host=HOST, port=PORT)
    app.run(host=HOST,debug=False, port=PORT)