from flask import Flask, jsonify , request , send_file, current_app, abort
from whisper_online import *
from faster_whisper import *
from mimetypes import guess_extension
import io , os , sys , tempfile, datetime
from pydub import AudioSegment , playback
from typing import IO, Iterable


RATE = 16000
HOST = '10.105.173.63'
PORT = 5000

src_lan = "ar"
tgt_lan = "ar"
model_size = 'large-v3'
model = FasterWhisperASR(lan=tgt_lan, modelsize=model_size,device='cuda')


def exportFile(file):
    seg:AudioSegment = AudioSegment.from_wav(file.stream)
    exported = seg.export('temp-wav-{}.wav'.format(getFormattedDate()))
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

@app.route('/transcript', methods = ['POST'])
def transcript():
    if('wav-file' in request.files):
        file = request.files['wav-file']
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