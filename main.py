from __future__ import unicode_literals
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv

import sys
import logging
import os
from typing import BinaryIO
import validators
import click
import yt_dlp as youtube_dl
from pydub import AudioSegment
import math

load_dotenv()


@click.group(chain=True)
def cli() -> None:
    pass


OPENAI_API_KEY = os.getenv('OPENAI_KEY')
logger = logging.getLogger(__name__)


def split_mp3(file_path: str, chunk_size_mb: int) -> int:
    logger.info("Splitting the mp3")
    # Load the MP3 file
    audio = AudioSegment.from_mp3(file_path)
    title = file_path.split('.')[0]

    # Calculate the chunk size in milliseconds
    chunk_size_ms = (chunk_size_mb * 1024 * 1024) / (audio.frame_rate * (audio.sample_width * 8) / 8.0 * audio.channels)
    chunk_size_ms = math.floor(chunk_size_ms * 1000)  # Convert to milliseconds

    expected_chunks = math.ceil(len(audio) / chunk_size_ms)
    existing_chunks = [fname for fname in os.listdir('.') if f'{title}_' in fname and fname.endswith('mp3')]

    if expected_chunks == len(existing_chunks):
        logger.info("Chunks already in place nothing to split")
        return len(existing_chunks)

    # Split the audio into chunks
    chunks = [audio[i:i + chunk_size_ms] for i in range(0, len(audio), chunk_size_ms)]
    print(chunks)

    # Export each chunk
    for i, chunk in enumerate(chunks):
        chunk.export(f"{title}_{i}.mp3", format="mp3")
        logger.info(f"Exported {title}_{i}.mp3")

    return len(chunks)


def transcribe(client: OpenAI, audio_file: BinaryIO) -> str | None:
    logger.info("Sending request to GPT for transcription")
    try:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )

        return transcription.text
    except OpenAIError as e:
        logger.error(f"OpenAI API error: {e}")


def text_to_speech(client: OpenAI, text: str, output_filename: str) -> None:
    logger.info("Sending text to create audio file")
    try:
        client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text
        ).stream_to_file(output_filename)
    except OpenAIError as e:
        logger.error(f"OpenAI API error: {e}")


def gpt_summarize(client: OpenAI, prompt: [dict[str, str]]) -> str | None:
    logger.info("Sending request to GPT for summary.")
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=prompt,
            temperature=0,
        )

        return response.choices[0].message.content.replace('\n', ' ').replace('\r', '')
    except OpenAIError as e:
        logger.error(f'OpenAI API error: {e}')


@click.command()
@click.option("--log", default='info')
@click.option('--clean', is_flag=True)
@click.option('--url', required=True)
@click.option('--tts', is_flag=True)
def summarize(log: str, url: str, clean: bool, tts: bool) -> None:
    log = log.upper()

    if log not in ["INFO", "ERROR", "DEBUG", "CRITICAL"]:
        logger.error("Invalid log level.")
        sys.exit(1)

    logging.basicConfig(level=log)

    if not validators.url(url):
        logger.error("Invalid Email.")
        sys.exit(1)

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': '%(title)s.%(ext)s',
        'logger': logger,
        'restrictfilenames': True
    }

    title: str

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        title = info['title']
        title = title.replace(' ', '_').replace('.', '')

    prev_files = [file for file in os.listdir() if title in file]

    if clean:
        for file in prev_files:
            logger.info(f"Removed previous file: {file}")
            os.remove(file)

    if not f'{title}.mp3' in os.listdir():
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    else:
        logger.info("Skipping download, file found locally")

    num_of_chunks = split_mp3(f'{title}.mp3', 200)

    content = []
    client = OpenAI(api_key=OPENAI_API_KEY)

    if not f'{title}.txt' in os.listdir():
        for c in range(num_of_chunks):
            if f'{title}_{c}.txt' in os.listdir():
                with (f'{title}_{c}.txt', 'r') as f:
                    logger.info("Reading from disk instead of API call")
                    content.append(f.read())
                    print(content)
                continue

            with open(f'{title}_{c}.mp3', "rb") as file:
                if transcription := transcribe(client, file):
                    prompt = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": f"Summarize the following text into bullet points: {transcription}."},
                    ]

                    if summary := gpt_summarize(client, prompt):
                        with open(f'{title}_{c}.txt', "w") as out:
                            out.write(summary)
                        content.append(summary)

        complete_transcript = ' '.join(content)

        prompt = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Summarize the following into an simple text: {complete_transcript}."},
        ]
        final_summary = gpt_summarize(client, prompt)

        with open(f'{title}.txt', 'w') as out:
            out.write(final_summary.replace(' - ', '\n- '))
            logger.info(f"Final summary written to: {title}.txt")
    else:
        logger.info("Nothing to do. Video already processed")

        with open(f'{title}.txt', 'r') as f:
            logger.info(f"Final summary read from : {title}.txt")
            complete_transcript = f.read()

    print(complete_transcript)

    if tts:
        text_to_speech(client, complete_transcript, f'{title}_transcript.mp3')


if __name__ == '__main__':
    cli.add_command(summarize)
    cli()

