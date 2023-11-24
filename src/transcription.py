import traceback
import numpy as np
import os
import sounddevice as sd
import tempfile
import wave
import webrtcvad
import whisper

"""
Record audio until speaking stops, then transcribe
"""
def transcribe(cancel, config=None):
    sample_rate = 16000
    frame_duration = 30
    silence_duration = config['silence_duration'] if config else 900

    vad = webrtcvad.Vad(3)
    buffer = []
    recording = []
    num_silent_frames = 0
    num_silence_frames = silence_duration // frame_duration
    try:
        with sd.InputStream(samplerate=sample_rate,
        channels=1, 
        dtype='int16',
        blocksize=sample_rate * frame_duration // 1000,
        device=config['recording_device_index'] if config else None,                        
        callback=lambda indata,
        frames,
        time, 
        status: buffer.extend(indata[:, 0])):
            while not cancel():
                if len(buffer) < sample_rate * frame_duration // 1000:
                    continue

                frame = buffer[:sample_rate * frame_duration // 1000]
                buffer = buffer[sample_rate * frame_duration // 1000:]

                is_speech = vad.is_speech(np.array(frame).tobytes(), sample_rate)
                if is_speech:
                    recording.extend(frame)
                    num_silent_frames = 0
                else:
                    if len(recording) > 0:
                        num_silent_frames += 1

                    if num_silent_frames >= num_silence_frames:
                        break


        if cancel():
            return ''
        
        audio_data = np.array(recording, dtype=np.int16)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            with wave.open(temp_file.name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data.tobytes())

        model_options = config['whisper_options']
        model = whisper.load_model(name=model_options['model'],
                                    device=model_options['device'])
        response = model.transcribe(audio=temp_file.name,
                                    language=model_options['language'],
                                    verbose=model_options['verbose'],
                                    initial_prompt=model_options['initial_prompt'],
                                    condition_on_previous_text=model_options['condition_on_previous_text'],
                                    temperature=model_options['temperature'],)
        
        os.remove(temp_file.name)
        
        if cancel():
            return ''

        result = response.get('text')
        
        return result.strip() if result else ''
            
    except Exception as e:
        traceback.print_exc()
