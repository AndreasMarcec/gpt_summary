# YouTube Audio Summarizer

This project provides a command-line tool for downloading YouTube videos, extracting their audio, splitting the audio into manageable chunks, transcribing the audio using OpenAI's Whisper API, summarizing the transcription using OpenAI's GPT-4, and optionally converting the summary back to speech.

## Features

- **Download YouTube Videos:** Download audio from YouTube videos in the best available quality.
- **Split Audio:** Split large audio files into smaller chunks to comply with OpenAI's API size limitations.
- **Transcribe Audio:** Transcribe the audio chunks using OpenAI's Whisper API.
- **Summarize Transcriptions:** Summarize the transcriptions into concise bullet points using OpenAI's GPT-4.
- **Text-to-Speech:** Convert the final summary back to speech using OpenAI's text-to-speech API.
- **Logging:** Comprehensive logging to help with debugging and monitoring the process.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7 or higher installed on your machine.
- A valid OpenAI API key.
- FFmpeg installed on your system (required by `yt_dlp` for audio processing).
- Required Python packages, which can be installed using the provided `requirements.txt` file.

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/YouTube-Audio-Summarizer.git
    cd YouTube-Audio-Summarizer
    ```

2. **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Set up your OpenAI API key:**

   Create a `.env` file in the root directory of the project and add your OpenAI API key:

    ```plaintext
    OPENAI_KEY=your_openai_api_key
    ```

4. **Ensure FFmpeg is installed:**

   Follow the instructions on the [FFmpeg website](https://ffmpeg.org/download.html) to install FFmpeg on your system.

## Usage

To use this tool, run the `summarize` command with the necessary options:

```bash
python main.py summarize --url <YouTube-URL> [--log <log-level>] [--clean] [--tts]
```

### Options

- `--url`: (Required) The URL of the YouTube video to process.
- `--log`: (Optional) Set the logging level. Options are `INFO`, `ERROR`, `DEBUG`, `CRITICAL`. Default is `INFO`.
- `--clean`: (Optional) Remove previously downloaded and processed files before starting.
- `--tts`: (Optional) Convert the final summary back to speech.

### Example

Download, transcribe, and summarize a YouTube video:

```bash
python main.py summarize --url https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

Download, transcribe, summarize, and convert the summary to speech:

```bash
python main.py summarize --url https://www.youtube.com/watch?v=dQw4w9WgXcQ --tts
```

## How It Works

1. **Download Audio:** The tool downloads the audio from the provided YouTube URL using `yt_dlp`.
2. **Split Audio:** The downloaded audio is split into smaller chunks if necessary.
3. **Transcription:** Each audio chunk is sent to OpenAI's Whisper API for transcription.
4. **Summarization:** The transcriptions are summarized into concise bullet points using OpenAI's GPT-4.
5. **Text-to-Speech (optional):** The final summary can be converted to speech using OpenAI's text-to-speech API.

## Logging

The tool provides comprehensive logging at different levels (`INFO`, `ERROR`, `DEBUG`, `CRITICAL`). By default, the logging level is set to `INFO`. You can change the logging level using the `--log` option.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes. Make sure to update the documentation as needed.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.


Feel free to open an issue or contact us if you have any questions or feedback.
