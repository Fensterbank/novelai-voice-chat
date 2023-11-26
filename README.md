# NovelAI Voice Chat

**NovelAI Voice Chat** is a small python tool to simulate a voice chat with an AI companion.  
It uses [OpenAI's local Whisper model](https://github.com/openai/whisper) to transcribe the spoken message and generates text and speech using [novelai-api](https://github.com/Aedial/novelai-api).

You need a valid [NovelAI](https://novelai.net/) subscription to use it.

The chat history is kept in JSON files including basic information, voice seeds and names.  
If more than five minutes passed between the last message, "director notes" like `{ 21 minutes later. It's Friday, 2023/11/24, 19:16. }` are added to better reflect the passage of time. 

The prompt for NovelAI is built by using `introduction`, `memory` and the last 2000 messages from the context file.  
This is highly experimental and I hope there is a better way to fully use the available token limit in the NovelAI subscription.

**Watch a small recording of the script in action**  
*(Please excuse my most German accent imaginable.)*  
[![Watch the video](/docs/demo-thumb.jpg)](https://storage.f-bit.software/f/cfb6d8b9b75a4d3dbb0b/)

## Hotkeys
There are a few hotkeys available.

| Hotkey | Function |
| -------- | --------  |
| L | Lists all audio devices. Useful if you have multiple microphones and speakers connected. The device id can be configured in `src/config.json` |
| Z | Speak to the AI and get a response. Probably the default use case. |
| X | Add a custom director note by speaking into the microphone without getting new AI output. Useful to add some story information like `It's a rainy day.` |
| C | Let the AI say something. Useful if your companion should take the initiative or if it should react to a director note you just added before. |
| D | Delete the last message. Useful to undo/redo one of your previous actions. Can be repeated. |
| P | Print the hotkeys to the console output. |

## Prerequisites
Before you can run this app, you'll need to have the following software installed:

- Git: [https://git-scm.com/downloads](https://git-scm.com/downloads)
- Python >=3.7: [https://www.python.org/downloads/](https://www.python.org/downloads/)
  - The [Whisper Python package](https://github.com/openai/whisper) is only compatible with Python versions >=3.7.
- ffmpeg

In my case, I also needed to install these packages to make `pip install` work:
```
sudo apt-get install portaudio19-dev
sudo apt-get install python3-tk python3-dev
```

## Installation
To set up and run the project, follow these steps:

### 1. Clone the repository:

```
git clone https://github.com/Fensterbank/novelai-voice-chat.git
cd novelai-voice-chat
```

### 2. Create a virtual environment and activate it:

```
python -m venv venv

# For Linux and macOS:
source venv/bin/activate

# For Windows:
venv\Scripts\activate
```

### 3. Install the required packages:

```
pip install -r requirements.txt
```

### 4. Setup your NovelAI credentials as environment variables


Open the ".env" file and add in your OpenAI API key:
```
NAI_USERNAME=<your_novelai_mail>
NAI_PASSWORD=<your_novelai_password>
```

### 6. Run the Python code with a context file:

```
python run.py --context spaceship.json
```

## Configuration Options

The project uses a configuration file to customize its behaviour. To set up the configuration, modify the `src\config.json` file:

```jsonc
{
    "whisper_options": {
        // tiny, base, small, medium, large. The bigger the model, the slower it is.
        "model": "small",
        "device": null,
        "language": "en",
        "temperature": 0.0,
        "initial_prompt": null,
        "condition_on_previous_text": true,
        "verbose": false
    },
    "novelai_options": {
        // you can define other standard presets
        "preset": "Carefree"
    },
    // change the hotkeys
    "hotkeys": {
        "record": "z",
        "instruct": "x",
        "ai_speak": "c",
        "list_devices": "l",
        "print_hotkeys": "h"
    },
    // define what info should be printed to the console
    "output": {
        "print_last_prompt": true,
        "print_ai_response": true,
        "print_transcription": true
    },
    // You may want to disable timestamps, since AI can get mad, if you don't talk to them for days.
    "add_timestamps": true,
    // waiting time in ms before recording stops and transcribing starts
    "silence_duration": 900,
    // the device ids for playback and recording
    "playback_device_index": 15,
    "recording_device_index": 14
}
```

## Known Issues

Following issues are known:

- **Numba Deprecation Warning**: A [numba depreciation warning](https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-of-object-mode-fall-back-behaviour-when-using-jit) is displayed. This is an issue with the Whisper Python package and will be fixed in a future release. The warning can be safely ignored.

- **FP16 Not Supported on CPU Warning**: A warning may show if you are running the local model on your CPU rather than a GPU using CUDA. This can be safely ignored.

- **Novel AI Settings**: I'm not sure, if I use the best settings for that approach. Especially with a small history, as always the start can be a bit bumpy.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
