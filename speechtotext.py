import speech_recognition as sr
from pydub import AudioSegment

def stt(audio_path=None):
    recognizer = sr.Recognizer()

    if audio_path:
        try:
            audio = AudioSegment.from_file(audio_path)
            wav_path = "temp.wav"
            audio.export(wav_path, format="wav")

            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)
                return text

        except Exception as e:
            return f"[Error] Could not process audio file: {e}"

    else:
        with sr.Microphone() as source:
            print("Recognizing speech...")
            audio_data = recognizer.listen(source)
            try:
                return recognizer.recognize_google(audio_data)
            except sr.UnknownValueError:
                return "[Error] Could not understand audio"
            except sr.RequestError as e:
                return f"[Error] API error: {e}"
