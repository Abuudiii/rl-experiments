"""Compose the Unreal re-render of a recorded drive (frames from scripts/unreal_render.py).

  python scripts/make_unreal_video.py RECORD_DIR FRAMES_DIR OUT.mp4 [--gif OUT.gif --gif-start S]

Same 2x2 layout as make_video.py: pilot view (Unreal, with the real 64x64 Donkey frame the agent drove
from inset), chase (Unreal), hawkeye (Unreal, fixed camera above the track) and the map with telemetry of
the recording. A caption under the panels says for the whole video that this is a re-render. One video
frame per recorded step, 20 fps. Needs ffmpeg on PATH.
"""
import argparse
import json
import pathlib
import subprocess

from PIL import Image, ImageDraw, ImageFont

W, H = 640, 480          # one panel (the Unreal frames are rendered at this size)
CAP = 84                 # caption bar under the panels (two lines, large enough to read in the 480 px GIF)
HZ = 20                  # the simulator sends one frame per 0.05 s of simulated time
BG, FG, DIM = (18, 18, 22), (235, 235, 235), (140, 140, 150)
ACCENT, TRAIL = (80, 220, 120), (240, 200, 60)
CAPTION = 'Re-render in Unreal of a recorded drive. The agent drove in the Donkey simulator; only the visuals are replayed here.'

parser = argparse.ArgumentParser()
parser.add_argument('record', type=pathlib.Path)
parser.add_argument('frames', type=pathlib.Path)
parser.add_argument('out', type=pathlib.Path)
parser.add_argument('--title', default='DreamerV3 on the Donkey Car simulator')
parser.add_argument('--gif', type=pathlib.Path, help='also write a GIF excerpt (fps 8, width 480) here')
parser.add_argument('--gif-start', type=float, default=0.0)
parser.add_argument('--gif-seconds', type=float, default=10.0)
args = parser.parse_args()


def font(size):
  """A scalable sans font, found through fontconfig; PIL's built-in bitmap font if there is none."""
  found = subprocess.run(['fc-match', '-f', '%{file}', 'sans'], capture_output=True, text=True).stdout
  return ImageFont.truetype(found, size) if found else ImageFont.load_default()


SMALL, MEDIUM, BIG = font(15), font(18), font(36)
rows = [json.loads(line) for line in (args.record / 'steps.jsonl').read_text().splitlines()]
# Keep the first episode only, so the video is one continuous drive (the frames cover exactly these).
end = next((i for i, row in enumerate(rows) if row['last']), len(rows) - 1)
rows = rows[:end + 1]
track = json.loads((args.record / 'track.json').read_text())

# Telemetry positions are in the sim's reduced units; node positions may be in world units (8x larger).
xs, zs = [r['x'] for r in rows], [r['z'] for r in rows]
span = max(max(xs) - min(xs), max(zs) - min(zs))
if track:
  tspan = max(max(p[0] for p in track) - min(p[0] for p in track), max(p[1] for p in track) - min(p[1] for p in track))
  scale = 1 / 8 if tspan > 4 * span else 1.0
  track = [(p[0] * scale, p[1] * scale) for p in track]
points = track + list(zip(xs, zs))
lo_x, hi_x = min(p[0] for p in points), max(p[0] for p in points)
lo_z, hi_z = min(p[1] for p in points), max(p[1] for p in points)
MAP_W, MAP_H = 280, 400   # map area inside the last panel
zoom = min(MAP_W / (hi_x - lo_x), MAP_H / (hi_z - lo_z))


def to_map(x, z):
  px = 18 + MAP_W / 2 + (x - (lo_x + hi_x) / 2) * zoom
  py = 52 + MAP_H / 2 - (z - (lo_z + hi_z) / 2) * zoom
  return px, py


def label(draw, text):
  draw.rectangle([0, 0, W, 26], fill=(0, 0, 0))
  draw.text((10, 4), text, font=SMALL, fill=FG)


def bar(draw, y, name, value, lo, hi, text):
  x0, x1 = 330, W - 80
  draw.text((x0, y), name, font=SMALL, fill=DIM)
  draw.text((x1 + 10, y + 18), text, font=SMALL, fill=FG)
  draw.rectangle([x0, y + 22, x1, y + 38], outline=DIM)
  zero = x0 + (0 - lo) / (hi - lo) * (x1 - x0)
  pos = x0 + (min(max(value, lo), hi) - lo) / (hi - lo) * (x1 - x0)
  draw.rectangle([min(zero, pos), y + 23, max(zero, pos), y + 37], fill=ACCENT)


def unreal(view, frame):
  im = Image.open(args.frames / view / f'{frame}.jpg').convert('RGB')
  return im if im.size == (W, H) else im.resize((W, H), Image.LANCZOS)


ffmpeg = subprocess.Popen([
    'ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{2 * W}x{2 * H + CAP}',
    '-r', str(HZ), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', str(args.out)],
    stdin=subprocess.PIPE)
trail = []
CAPTION_FONT = font(30)
caption_lines = [CAPTION[:CAPTION.index('. ') + 1], CAPTION[CAPTION.index('. ') + 2:]]
caption_w = [ImageDraw.Draw(Image.new('RGB', (1, 1))).textlength(t, font=CAPTION_FONT) for t in caption_lines]
for index, row in enumerate(rows):
  canvas = Image.new('RGB', (2 * W, 2 * H + CAP), BG)

  panel = unreal('pilot', row['frame'])
  draw = ImageDraw.Draw(panel)
  label(draw, 'PILOT VIEW  (Unreal re-render from the car\'s camera position)')
  # Exactly what the agent received: the Donkey camera frame resized to 64x64 with PIL BILINEAR.
  donkey = Image.open(args.record / f"pilot_{row['frame']}.jpg").convert('RGB')
  agent = donkey.resize((64, 64), Image.BILINEAR).resize((128, 128), Image.NEAREST)
  x0, y0 = W - 138, H - 138
  panel.paste(agent, (x0, y0))
  draw.rectangle([x0 - 1, y0 - 1, x0 + 128, y0 + 128], outline=FG)
  text = 'agent input 64x64 (real Donkey simulator frame)'
  tw = draw.textlength(text, font=SMALL)
  draw.rectangle([W - tw - 18, y0 - 26, W - 4, y0 - 5], fill=(0, 0, 0))
  draw.text((W - tw - 11, y0 - 24), text, font=SMALL, fill=FG)
  canvas.paste(panel, (0, 0))

  panel = unreal('chase', row['frame'])
  label(ImageDraw.Draw(panel), 'CHASE  (Unreal re-render)')
  canvas.paste(panel, (W, 0))

  panel = unreal('hawk', row['frame'])
  label(ImageDraw.Draw(panel), 'HAWKEYE  (Unreal re-render, fixed camera above the track)')
  canvas.paste(panel, (0, H))

  panel = Image.new('RGB', (W, H), BG)
  draw = ImageDraw.Draw(panel)
  label(draw, 'MAP AND TELEMETRY  (recorded)  ' + args.title)
  if track:
    draw.line([to_map(*p) for p in track + track[:1]], fill=(90, 90, 100), width=9)
  trail.append(to_map(row['x'], row['z']))
  if len(trail) > 1:
    draw.line(trail, fill=TRAIL, width=2)
  x, y = trail[-1]
  draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=ACCENT, outline=FG)
  draw.text((330, 50), f"lap {min(int(row['laps']) + 1, max(1, round(rows[-1]['laps'])))}", font=BIG, fill=FG)
  draw.text((330, 94), f"{index / HZ:5.1f} s   {row['laps']:.2f} laps", font=MEDIUM, fill=DIM)
  bar(draw, 150, 'steering', row['steer'], -1, 1, f"{row['steer']:+.2f}")
  bar(draw, 215, 'throttle', row['throttle'], 0, 1, f"{row['throttle']:.2f}")
  bar(draw, 280, 'speed', row['speed'], 0, 5, f"{row['speed']:.1f}")
  bar(draw, 345, 'off centre', row['cte'], -3, 3, f"{row['cte']:+.1f}")
  canvas.paste(panel, (W, H))

  draw = ImageDraw.Draw(canvas)
  for k, (text, tw) in enumerate(zip(caption_lines, caption_w)):
    draw.text(((2 * W - tw) / 2, 2 * H + 6 + 38 * k), text, font=CAPTION_FONT, fill=(255, 220, 90))

  ffmpeg.stdin.write(canvas.tobytes())
ffmpeg.stdin.close()
if ffmpeg.wait():
  raise SystemExit('ffmpeg failed')
print(f'wrote {args.out}: {len(rows)} frames, {len(rows) / HZ:.1f} s, {rows[-1]["laps"]:.2f} laps')

if args.gif:
  subprocess.run([
      'ffmpeg', '-v', 'error', '-y', '-ss', str(args.gif_start), '-t', str(args.gif_seconds), '-i', str(args.out),
      '-vf', 'fps=8,scale=480:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse', str(args.gif)], check=True)
  print(f'wrote {args.gif}: {args.gif.stat().st_size / 1e6:.2f} MB')
