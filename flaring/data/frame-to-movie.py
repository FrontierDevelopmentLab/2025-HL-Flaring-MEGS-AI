import os
import glob
import imageio.v2 as imageio
from tqdm import tqdm

frames_base = "/mnt/data2/frames"
output_dir = "/mnt/data/movies/"
os.makedirs(output_dir, exist_ok=True)

wavelengths = ['94', '131', '171', '193', '211', '304']
fps = 10  # frames per second
duration = 1 / fps  # seconds per frame

#Generate GIF for each wavelength
for wl in wavelengths:
    frame_dir = os.path.join(frames_base, wl)
    images = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
    if not images:
        print(f"No images found for wavelength {wl}")
        continue
    output_path = os.path.join(output_dir, f"aia_{wl}.mp4")
    print(f"Creating MP4 movies for AIA {wl} Å with {len(images)} frames...")
    with imageio.get_writer(output_path, format='ffmpeg', fps=fps) as writer: ##remove codec='libx264', mp4 doesn't play
        for img_path in tqdm(images, desc=f"AIA {wl} Å", ncols=100):
            img = imageio.imread(img_path)
            writer.append_data(img)
    print(f"Saved: {output_path}")