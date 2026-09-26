import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "output" / "DDK001_test_video_narrated.mp4"
SRT = ROOT / "output" / "DDK001_subtitles.srt"
OUTPUT = ROOT / "output" / "DDK001_test_video_subtitled.mp4"


def main():
    if not VIDEO.exists():
        raise RuntimeError(f"Missing narrated video: {VIDEO}")
    if not SRT.exists():
        raise RuntimeError(f"Missing subtitles: {SRT}")

    # FFmpeg's subtitles filter burns readable captions into the video.
    subtitle_path = str(SRT).replace("\\", "/").replace(":", "\\:")
    vf = (
        f"subtitles='{subtitle_path}':"
        "force_style='FontName=DejaVu Sans,FontSize=28,"
        "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
        "BorderStyle=1,Outline=2,Shadow=1,Alignment=2,MarginV=35'"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(VIDEO),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "22",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(OUTPUT),
    ]

    subprocess.run(cmd, check=True)
    print(f"Created subtitled test video: {OUTPUT}")


if __name__ == "__main__":
    main()
