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
model_size = 'medium'
model = WhisperModel(model_size)

asr = FasterWhisperASR(lan=tgt_lan, modelsize=model_size,device='cuda')
online = OnlineASRProcessor(asr)

def exportFile(file:IO[bytes]) -> str:
    instance_folder = current_app.instance_path
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'uploads')
    folder = os.path.join(instance_folder,upload_folder)
    if(not os.path.exists(folder)):
        os.makedirs(folder)

    file_dest = os.path.join(folder,'recorded_audio.wav')
    file_bytes_len = len(file.read())
    sample_width = 1 if file_bytes_len % (1 * 1) == 0 else 2
    sound:AudioSegment = AudioSegment(file.read(),channels=1,frame_rate=RATE,sample_width=sample_width)
    playback.play(sound)
    sound.export(file_dest,format='wav')
    # print(f'{sound}', file=sys.stderr, flush=True)
    return file_dest


def handleSegments(segments:list):
    o = []
    for segment in segments:
        for word in segment.words:
            # not stripping the spaces -- should not be merged with them!
            w = word.word
            t = (word.start, word.end, w)
            o.append(t)
    return o


def cleanFilesCache():
    pass

def wavToText(file_dest:str) -> str:
    try:
        model = FasterWhisperASR(lan=tgt_lan, modelsize=model_size,device='cuda')
        segments = model.transcribe(file_dest);
        # print('segments {}'.format(segments))
        return segments;
    except Exception as e:
        print(f"Some Error! {e} happened while generating text for {file_dest}")

app = Flask(__name__)

@app.route('/transcript', methods = ['POST'])
def transcript():
    if('wav-file' in request.files):
        file = request.files['wav-file']
        print('creating segment')
        seg:AudioSegment = AudioSegment.from_wav(file.stream)
        print('creating file')
        file = seg.export('temp-wav-{}.wav'.format(datetime.datetime.now().date()))
        print('creating text results from file')
        text_result = wavToText(file.name);
        file.close()
        os.remove(file.name)
        # return
        print(f'result: {text_result}')
    return jsonify({'transcript': text_result})

if __name__ == '__main__':
    # from waitress import serve
    # serve(app, host=HOST, port=PORT)
    app.run(host=HOST,debug=False, port=PORT)