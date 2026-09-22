import sys
from pathlib import Path

import PIL
import PIL.ImageFont

# TRDG versions that still call FreeTypeFont.getsize() are incompatible
# with Pillow 10+, where getsize() was removed.
#
# IMPORTANT:
# - The old TRDG code uses getsize()[1] as the text canvas height.
# - Pillow's getbbox() returns (left, top, right, bottom).
# - Therefore the compatible height is `bbox[3]`, NOT `bbox[3] - bbox[1]`.
# - Do NOT move ImageDraw.text(). TRDG already draws at (0, 0) correctly.
if not hasattr(PIL.ImageFont.FreeTypeFont, "getsize"):
    def getsize(self, text, *args, **kwargs):
        bbox = self.getbbox(text)
        if bbox is None:
            return (0, 0)

        width = round(self.getlength(text))
        height = bbox[3]

        return (width, height)

    PIL.ImageFont.FreeTypeFont.getsize = getsize

from trdg.run import main


if __name__ == "__main__":
    # Keep all paths relative to the project directory from which you run
    # this script.
    sys.argv = [
        "trdg",
        "-c", "50000",
        "-l", "ru",
        "-f", "32",
        "-w", "1",
        "-t", "8",
        "-fd", r"dataset\ocr_training\fonts",
        "-dt", r"dataset\ocr_training\lexicon_dates.txt",
        "--output_dir", r"dataset\ocr_training\dated_synth_crops",
        "-k", "2",
        "-rk",
        "-bl", "1",
        "-rbl",
        "-tc", "#111111,#222222,#4A0E17,#1A2B4C,#8B6508",
        "-b", "1",
        "-d", "1",
    ]

    print(f"[*] Pillow: {PIL.__version__}")
    print("[*] Starting TRDG generation...")
    main()
    print("[+] Generation finished.")