import os
import trdg

# Find where TRDG is installed in your virtual environment
trdg_path = os.path.dirname(trdg.__file__)
init_file = os.path.join(trdg_path, '__init__.py')

# The code we will inject to resurrect the deleted 'getsize' command
patch_code = """
# --- Pillow 10+ Compatibility Patch ---
import PIL.ImageFont
if not hasattr(PIL.ImageFont.FreeTypeFont, 'getsize'):
    def _getsize(self, text, *args, **kwargs):
        bbox = self.getbbox(text, *args, **kwargs)
        # Returns (width, height) by calculating the bounding box coordinates
        return (bbox[2] - bbox[0], bbox[3] - bbox[1]) if bbox else (0, 0)
    
    PIL.ImageFont.FreeTypeFont.getsize = _getsize
    PIL.ImageFont.ImageFont.getsize = _getsize
"""

# Append the patch to TRDG's initialization file
with open(init_file, 'r') as f:
    content = f.read()

if "Pillow 10+ Compatibility Patch" not in content:
    with open(init_file, 'a') as f:
        f.write(patch_code)
    print("[+] Successfully injected Pillow hotfix into TRDG!")
else:
    print("[*] TRDG is already patched.")