import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv

import whisper
from whisper.utils import get_writer
model = whisper.load_model('base')

#Creating the audio file
freq = 44100
duration = 5

recording = sd.rec(int (duration*freq), samplerate=freq, channels=2)
sd.wait()

write("recording0.wav", freq, recording)
wv.write("recording1.wav", recording, freq, sampwidth=2)

#Importing audio file into whisper
def get_transcribe(audio:'recording1.wav', language: str = 'en'):
    return model.transcribe(audio = audio, language=language, verbose = True)

#Save the file
def save_file(results, format = 'txt'):
    writer = get_writer(format, './output/')
    writer(results, f'transcribe.{format}')
