import os
import subprocess
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def extract_audio_from_video(video_path: str, audio_output_path: str):
    """
    Extracts audio from the given video file and saves it to audio_output_path.
    This example uses ffmpeg shell calls.
    """
    command = [
        "ffmpeg",
        "-i", video_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        audio_output_path
    ]
    try:
        subprocess.run(command, check=True)
        logger.info(f"Audio extracted: {audio_output_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to extract audio from {video_path}: {e}")

