# Console version of the image face recognizer: python scripts/ImageIdentify.py [image]
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'gui'))
from faceHandler import loadKnownFaces, analyzeImage

known, skipped = loadKnownFaces()
for name in skipped:
    print('No face found in ' + name)

# load test image
path = sys.argv[1] if len(sys.argv) > 1 else ROOT / 'img' / 'groups' / 'vladimir-putin-bashar-al-asad-y-barack-obama-1.jpg'
result, names, savedPath = analyzeImage(path, known)

print('Faces:', ', '.join(name or 'Unknown' for name in names) or 'none')
print('Saved in', savedPath)

# display image
Image.fromarray(result).show()
