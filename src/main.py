import json
import os
import threading
import asyncio
import datetime
import argparse
import soundfile as sf
from pathlib import Path
import sounddevice as sd
from typing import List, Optional
from novelai_api.BanList import BanList
from novelai_api.BiasGroup import BiasGroup
from novelai_api.GlobalSettings import GlobalSettings
from novelai_api.Preset import Model, Preset
from novelai_api.Tokenizer import Tokenizer
from novelai_api.utils import b64_to_tokens
from boilerplate import API
from transcription import transcribe
from pynput import keyboard

parser = argparse.ArgumentParser()
parser.add_argument('--context', help='Path to the context file')
args = parser.parse_args()

# Retrieve the value of --context parameter
context_file = args.context

if context_file is None:
  raise ValueError("context_file is not defined")

class ResultThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super(ResultThread, self).__init__(*args, **kwargs)
        self.result = None
        self.stop_transcription = False

    def run(self):
        self.result = self._target(*self._args, cancel=lambda: self.stop_transcription, **self._kwargs)
        
    def stop(self):
        self.stop_transcription = True
        
def load_config_with_defaults():
    default_config = {
        'whisper_options': {
            'model': 'base',
            'device': None,
            'language': None,
            'temperature': 0.0,
            'initial_prompt': None,
            'condition_on_previous_text': True,
            'verbose': False
        },
        'novelai_options': {
            'preset': 'Carefree',
        },
        'hotkeys': {
            'record': 'z',
            'instruct': 'x',
            'ai_speak': 'c',
            'delete_last_action': 'del',
            'list_devices': 'l',
            'print_hotkeys': 'h'
        },
        'output': {
            'print_last_prompt': True,
            'print_ai_response': True,
            'print_transcription': True
        },
        'add_timestamps': True,
        'silence_duration': 900,
        'playback_device_index': 0,
        'recording_device_index': 0
    }

    config_path = os.path.join('src', 'config.json')
    if os.path.isfile(config_path):
        with open(config_path, 'r') as config_file:
            user_config = json.load(config_file)
            for key, value in user_config.items():
                if key in default_config and value is not None:
                    default_config[key] = value

    return default_config

def get_time_difference(last_message_time: datetime.datetime) -> str:
    """
    Calculates the time difference between the current time and the given last message time.

    Args:
        last_message_time (datetime.datetime): The timestamp of the last message.

    Returns:
        str: A string representation of the time difference in days, hours, or minutes.
             Returns None if the time difference is less than or equal to 5 minutes.
    """
    current_time = datetime.datetime.now(datetime.timezone.utc)
    time_difference = current_time - last_message_time

    if time_difference.total_seconds() > 300:
        if time_difference.days > 0:
            time_difference_text = f"{time_difference.days} days"
        elif time_difference.seconds >= 3600:
            hours = time_difference.seconds // 3600
            time_difference_text = f"{hours} hours"
        elif time_difference.seconds >= 60:
            minutes = time_difference.seconds // 60
            time_difference_text = f"{minutes} minutes"

        return time_difference_text
    else:
        return None

def add_director_note_if_necessary(context: dict):
  formatted_date = datetime.datetime.now().strftime("%A, %Y/%m/%d, %H:%M")

  last_message = context['messages'][-1]
  last_message_date_string = last_message.get('date')
  # if last_message_date_string is null, this is the first user submitted message, so we add a director note without a time difference
  if last_message.get('date') is None:
    context['messages'].append({
      'sender': 'director',
      'date': datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'text': f" It's {formatted_date}."
    })
    return

  last_message_time = datetime.datetime.fromisoformat(last_message_date_string.replace('Z', '+00:00')).astimezone(datetime.timezone.utc)
  time_difference_text = get_time_difference(last_message_time)

  if time_difference_text:
    context['messages'].append({
      'sender': 'director',
      'date': datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'text': f"{time_difference_text} later. It's {formatted_date}."
    })

def build_prompt(context: dict) -> str:
  prompt = ''
  # add introduction if its not None, empty or whitespace string
  if context['introduction'] and context['introduction'].strip():
    prompt += context['introduction'] + '\n'
  # add memory if its not None, empty or whitespace string
  if context['memory'] and context['memory'].strip():
    prompt += context['memory'] + '\n'
  
  ai_name = context['ai_name']
  user_name = context['user_name']

  last_2000_messages = context["messages"][-2000:]
  
  for i, message in enumerate(last_2000_messages):
    sender = message.get('sender')
    text = message.get('text', '')

    if sender == 'ai':
      prompt += '\n' + f"{ai_name}:\n{text}"
    elif sender == 'user':
      prompt += '\n' + f"{user_name}:\n{text}"
    elif sender == 'director':
      prompt += '\n' + f"({text})"

    # Add author's note as the tenth last element
    if i == len(last_2000_messages) - 10 and context['authors_note'] and context['authors_note'].strip():
      prompt += '\n' + f"[{context['authors_note']}]"

  prompt += '\n'
  return prompt

async def generate_response(config: dict, api, prompt: str, model: Model, preset: Preset, global_settings: GlobalSettings, bad_words: Optional[BanList], bias_groups: List[BiasGroup], module, stop_sequence: List[str], context: dict) -> str:
  prompt = Tokenizer.encode(model, prompt)
  print('Story Size: ' + str(len(prompt)) + ' tokens.')
  
  gen = await api.high_level.generate(
    prompt,
    model,
    preset,
    global_settings,
    bad_words=bad_words,
    biases=bias_groups,
    prefix=module,
    stop_sequences=stop_sequence,
  )

  response = Tokenizer.decode(model, b64_to_tokens(gen['output']))
  response = response.strip()
  if config['output']['print_ai_response']:
      print('=> ' + response)
  return response

async def generate_voice(api, config: dict, context: dict, text: str):
  tts_file = "tts.mp3"
  d = Path("temp")
  d.mkdir(exist_ok=True)

  seed = context["voice_seed"]
  opus = False # for getting mp3
  version = "v2"

  tts = await api.low_level.generate_voice(text, seed, -1, opus, version)
  with open(d / tts_file, "wb") as f:
      f.write(tts)

  print("Playing AI response...")
  # play the audio
  play_audio(d / tts_file, device_index=config["playback_device_index"])
  print_hotkeys(config['hotkeys'])

def play_audio(file_path, device_index=None):
    # Read the audio file
    audio, samplerate = sf.read(file_path)

    # Play the audio using sounddevice
    sd.play(audio, samplerate=samplerate, device=device_index)
    sd.wait()

def list_audio_devices():
    print(sd.query_devices())
def save_context(context: dict):
  with open(context_file, 'w') as file:
    file.write(json.dumps(context))

def update_context(context: dict, response: str):
  current_time = datetime.datetime.now(datetime.timezone.utc)

  context['messages'].append({
    'date': current_time.isoformat(),
    'sender': 'ai',
    'text': response
  })

  save_context(context)

async def get_novelai_response(config: dict, context: dict, prompt: str):
  async with API() as api_handler:
    api = api_handler.api
    model = Model.Kayra

    preset = Preset.from_official(model, config['novelai_options']['preset'])
    preset.min_length = 1
    preset.max_length = 20
    if config['output']['print_last_prompt']:
      print(prompt)

    global_settings = GlobalSettings(num_logprobs=GlobalSettings.NO_LOGPROBS)
    global_settings.rep_pen_whitelist = True
    global_settings.generate_until_sentence = True

    bad_words: Optional[BanList] = BanList('[', ']', '(', ')', '\n', '<3', f"{context['user_name']}:")
    bias_groups: List[BiasGroup] = []

    module = None
    stop_sequence = ['\n']

    response = await generate_response(config, api, prompt, model, preset, global_settings, bad_words, bias_groups, module, stop_sequence, context)
    await generate_voice(api, config, context, response)
    return response

# default user input, adds director notes if necessary, build prompt and fetch from novelai
async def perform_input(config: dict, input_string: str):
    if config['output']['print_transcription']:
      print('Transcription: ' + input_string)

    with open(context_file, 'r') as file:
        context = json.load(file)

    if config['add_timestamps']:
      add_director_note_if_necessary(context)

    user_input = input_string

    context['messages'].append({
        'date': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'sender': 'user',
        'text': user_input
    })

    prompt = build_prompt(context)

    # we want an ai answer
    prompt += context['ai_name'] + ':\n'
    response = await get_novelai_response(config, context, prompt)

    update_context(context, response)

async def perform_ai_speak(config: dict):
    print('Performing ai speak...')

    with open(context_file, 'r') as file:
        context = json.load(file)
        
    if config['add_timestamps']:
      add_director_note_if_necessary(context)

    prompt = build_prompt(context)

    # we want an ai answer
    prompt += context['ai_name'] + ':\n'
    response = await get_novelai_response(config, context, prompt)

    update_context(context, response)

def add_instruct_and_save(input_string: str):
  print('Added instruct: ' + input_string)
  with open(context_file, 'r') as file:
      context = json.load(file)

  context['messages'].append({
    'sender': 'director',
    'date': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'text': input_string
  })
  save_context(context)

# let the user speak to the ai and add it including the ai's response to the context
def on_record():
    print('Speak to the AI...')
    recording_thread = ResultThread(target=transcribe, args=(), kwargs={'config': config})
    recording_thread.start()
    
    recording_thread.join()

    transcribed_text = recording_thread.result

    # Run the async function in an event loop
    asyncio.run(perform_input(config, transcribed_text))

# let the ai speak by itself. Is useful after having added a custom instruct or if the ai should go into the initiative.
def on_ai_speak():
    print('AI speaking...')

    asyncio.run(perform_ai_speak(config))

# let the user record an instruct message which will be placed at the end of the context without getting any response from the ai
def on_instruct(hotkeys: dict):
    print('Record a note...')

    recording_thread = ResultThread(target=transcribe, args=(), kwargs={'config': config})
    recording_thread.start()
    
    recording_thread.join()

    transcribed_text = recording_thread.result

    add_instruct_and_save(transcribed_text)
    print_hotkeys(hotkeys)

# delete the last message from the context
def delete_last_action():
  with open(context_file, 'r') as file:
      context = json.load(file)

  last_message = context['messages'][-1]      
  context['messages'].pop()
  save_context(context)
  print('Deleted last message: ' + last_message['text'])

def format_keystrokes(key_string):
    return '+'.join(word.capitalize() for word in key_string.split('+'))

def print_hotkeys(hotkeys: dict):
    print(f'Hotkeys:\n{format_keystrokes(hotkeys["list_devices"])} to list all audio devices for use in config.\n{format_keystrokes(hotkeys["record"])} to speak to the ai and get response.\n{format_keystrokes(hotkeys["instruct"])} to add a note or instruct without getting ai response.\n{format_keystrokes(hotkeys["ai_speak"])} to let the ai speak.\n{format_keystrokes(hotkeys["delete_last_action"])} to delete the last action.\n{format_keystrokes(hotkeys["print_hotkeys"])} to show this help.\nPress ESC to quit.')


def on_key_release(key):
  if key == keyboard.Key.esc:
    # Stop the listener if the 'esc' key is pressed
    return False
  else:
    hotkey = str(key).replace("'", "")
    hotkey_actions = {
      hotkeys['list_devices']: list_audio_devices,
      hotkeys['record']: on_record,
      hotkeys['instruct']: lambda: on_instruct(hotkeys),
      hotkeys['ai_speak']: on_ai_speak,
      hotkeys['delete_last_action']: delete_last_action,
      hotkeys['print_hotkeys']: lambda: print_hotkeys(hotkeys)
    }
    if hotkey in hotkey_actions:
      hotkey_actions[hotkey]()

# Main script
config = load_config_with_defaults()
hotkeys = config['hotkeys']
print_hotkeys(hotkeys)

# Start the listener
with keyboard.Listener(on_release=on_key_release) as listener:
  # Keep the script running
  listener.join()

