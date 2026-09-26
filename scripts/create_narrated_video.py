import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "output" / "DDK001_test_video.mp4"
AUDIO = ROOT / "output" / "DDK001_test_narration.mp3"
OUTPUT = ROOT / "output" / "DDK001_test_video_narrated.mp4"


def main():
    if not VIDEO.exists():
        raise RuntimeError(f"Missing video: {VIDEO}")
    if not AUDIO.exists():
        raise RuntimeError(f"Missing narration: {AUDIO}")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(VIDEO),
        "-i", str(AUDIO),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "128k",
        "-af", "apad",
        "-t", "12",
        "-shortest",
        "-movflags", "+faststart",
        str(OUTPUT),
    ]

    subprocess.run(cmd, check=True)
    print(f"Created narrated test video: {OUTPUT}")


if __name__ == "__main__":
    main()
