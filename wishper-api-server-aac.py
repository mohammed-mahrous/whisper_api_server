from flask import Flask, jsonify , request
from whisper_online import *
from faster_whisper import *
import os , datetime
from pydub import AudioSegment


RATE = 16000
HOST = 'localhost'
PORT = 5000

src_lan = "ar"
tgt_lan = "ar"
model_size = 'large-v3'
model = FasterWhisperASR(lan=tgt_lan, modelsize=model_size,device='cuda')


def exportFile(file):
    seg:AudioSegment = AudioSegment.from_file(file.stream, 'aac')
    exported = seg.export('temp-file-{}.mp3'.format(getFormattedDate()),format="wav", parameters=["-ar", "16000", "-ac", "1"])

    return exported


def handleSegments(segments:list) -> str:
    o = []
    for segment in segments:
        o.append(segment.text)
    text = " \r\n\r\n ".join([txt for txt in o])
    return text


def cleanFilesCache():
    pass

def wavToText(file_dest:str) -> str:
    try:
        segments = model.transcribe(file_dest);
        if(segments):
            return handleSegments(segments);
        return '';
    except Exception as e:
        print(f"Some Error! {e} happened while generating text for {file_dest}")

app = Flask(__name__)

def getFormattedDate() -> str:
    now = datetime.datetime.now()
    return f'{now.day}-{now.hour}-{now.minute}-{now.second}'

@app.route('/transcript/aac', methods = ['POST'])
def transcript():
    print(request)
    print(request.files)
    for key in request.files:
        print(f'{key}: -> {request.files[key]}')
        
    if('file' in request.files):
        file = request.files['file']
        exported = exportFile(file=file)
        text_result = wavToText(exported.name);
        exported.close()
        os.remove(exported.name)
        print(f'result: {text_result}')
    return jsonify({'transcript': text_result})

if __name__ == '__main__':
    # from waitress import serve
    # serve(app, host=HOST, port=PORT)
    app.run(host=HOST,debug=False, port=PORT)