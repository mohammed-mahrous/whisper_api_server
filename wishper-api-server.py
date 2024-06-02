from flask import Flask, jsonify , request , send_file, current_app, abort
from whisper_online import *
from faster_whisper import *
from mimetypes import guess_extension
import io , os , sys , tempfile, datetime
from pydub import AudioSegment , playback
from typing import IO, Iterable


RATE = 44100
src_lan = "ar"
tgt_lan = "ar"
model_size = 'tiny'
model = WhisperModel(model_size)
# asr = FasterWhisperASR(lan=tgt_lan, modelsize=model_size,device='auto')
# online = OnlineASRProcessor(asr)

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


def cleanFilesCache():
    pass

def wavToText(file_dest:str) -> str:
    import whisper
    try:
        model = whisper.load_model('medium')
        result = model.transcribe(file_dest,language='ar');
        return result['text'];
    except Exception as e:
        print(f"Some Error! {e} happened while generating text for {file_dest}")

app = Flask(__name__)

@app.route('/transcript', methods = ['POST'])
def transcript():
    if('wav-file' in request.files):
        file = request.files['wav-file']
        file_bytes_len = len(file.stream.read())
        print('len = {}'.format(file_bytes_len))
        seg:AudioSegment = AudioSegment.from_wav(file.stream)
        file = seg.export('temp-wav-{}.wav'.format(datetime.datetime.now().date()))
        text_result = wavToText(file.name);
        file.close()
        os.remove(file.name)
        print(f'result: {text_result}')
    return jsonify({'transcript': text_result})

if __name__ == '__main__':
    from waitress import serve
    serve(app, host="10.105.173.63", port=8080)
    # app.run(host='localhost',debug=True, port=5000)