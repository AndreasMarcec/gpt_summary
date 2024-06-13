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

import tiktoken

load_dotenv()

# Test: Transcribing https://www.youtube.com/watch?v=Wpf7G7E1Jzw
#       and using Tiktoken with gpt-4o results into 14267 tokens
#       so we have 14267 tokens / ~52min = ~ 275 token/min
#       now we can use to base our estimation of the input price

@click.group(chain=True)
def cli() -> None:
    pass


OPENAI_API_KEY = os.getenv('OPENAI_KEY')
logger = logging.getLogger(__name__)


def get_num_tokens(very_long_string: str):
    enc = tiktoken.encoding_for_model("gpt-4o")
    return len(enc.encode(very_long_string))


# TODO Whipser has a limit of 25MB per request -> Either find better way to split or split again if to large
def split_mp3(file_path: str, chunk_len_in_s=1000) -> int:
    logger.info("Splitting the mp3")
    # Load the MP3 file
    audio = AudioSegment.from_mp3(file_path)
    title = file_path.split('.')[0]

    # Calculate the chunk size in milliseconds
    chunk_size_ms = chunk_len_in_s * 1000  # Convert to milliseconds

    expected_chunks = math.ceil(len(audio) / chunk_size_ms)
    existing_chunks = [fname for fname in os.listdir('.') if f'{title}_' in fname and fname.endswith('mp3')]

    if expected_chunks == len(existing_chunks):
        logger.info("Chunks already in place nothing to split")
        return len(existing_chunks)

    # Split the audio into chunks
    chunks = [audio[i:i + chunk_size_ms] for i in range(0, len(audio), chunk_size_ms)]

    # Export each chunk
    for i, chunk in enumerate(chunks):
        chunk.export(f"{title}_{i}.mp3", format="mp3")
        logger.info(f"Exported {title}_{i}.mp3")

    return len(chunks)


def transcribe(client: OpenAI, audio_file: BinaryIO) -> str | None:
    logger.info(f"Transcribing {audio_file.name}")
    try:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )

        return transcription.text
    except OpenAIError as e:
        logger.error(f"OpenAI API error: {e}")


def transcribe_chunks(client: OpenAI, num_of_chunks, title) -> str:
    logger.info("Starting to transcribe all chunks")
    combined_transcription = []

    for i in range(num_of_chunks):
        with open(f'{title}_{i}.mp3', "rb") as audio_file:
            combined_transcription.append(transcribe(client, audio_file))

    return " ".join(combined_transcription)


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


def estimate_transcription(playtime: int):
    print(playtime)
    print(math.ceil(playtime/60))
    print(float(os.getenv("WHISPER_PRICE")))
    return math.ceil(playtime/60) * float(os.getenv("WHISPER_PRICE"))


@click.command()
@click.option('--url', required=True)
def estimate_cost(url: str):
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

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        duration = info["duration"]
        logger.info(f"Duration: {duration}")

        print(estimate_transcription(duration))



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

    num_of_chunks = split_mp3(f'{title}.mp3')

    client = OpenAI(api_key=OPENAI_API_KEY)
    combined_transcription = transcribe_chunks(client, num_of_chunks, title)


    content = []

    if not f'{title}.txt' in os.listdir():
        for c in range(num_of_chunks):
            if f'{title}_{c}.txt' in os.listdir():
                with (f'{title}_{c}.txt', 'r') as f:
                    logger.info("Reading from disk instead of API call")
                    content.append(f.read())
                    print(content)
                continue

            # TODO Refactor the following lines and use
            #      transcribe_chunks(...) instead
            with open(f'{title}_{c}.mp3', "rb") as file:
                if transcription := transcribe(client, file):
                    prompt = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user",
                         "content": f"Summarize the following text into bullet points: {transcription}."},
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
    cli.add_command(estimate_cost)
    cli()
    # split_mp3("Life_on_House_and_Portraying_Dr._Taub_Peter_Jacobson-[MMIrp0F4_sE].mp3")
