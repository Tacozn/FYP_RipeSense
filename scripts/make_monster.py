from PIL import Image
import numpy as np
import os

# 1. Create a massive 5000x5000 image (approx 75MB in RAM when raw)
print(" Concocting a heavy image...")
array = np.random.rand(5000, 5000, 3) * 255
img = Image.fromarray(array.astype('uint8')).convert('RGB')

# 2. Save it
img.save("static/monster_image.jpg")
print(f" 'monster_image.jpg' created! Size: {img.size}")
print(f" File size on disk: {os.path.getsize('static/monster_image.jpg')/1024/1024:.2f} MB")