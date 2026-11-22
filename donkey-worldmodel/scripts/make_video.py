"""Compose the demo video from a folder written by envs/donkey_record.py.

  python scripts/make_video.py RECORD_DIR OUT.mp4 [--title TEXT]

Four panels: pilot camera (with the 64x64 image the agent actually receives inset), chase camera,
top-down map with the car's path, and telemetry. Needs ffmpeg on PATH.
"""
import argparse
import json
import pathlib
import subprocess

from PIL import Image, ImageDraw, ImageFont

W, H = 480, 360          # one panel
HZ = 20                  # the simulator sends one frame per 0.05 s of simulated time
BG, FG, DIM = (18, 18, 22), (235, 235, 235), (140, 140, 150)
ACCENT, TRAIL = (80, 220, 120), (240, 200, 60)

parser = argparse.ArgumentParser()
parser.add_argument('record', type=pathlib.Path)
parser.add_argument('out', type=pathlib.Path)
parser.add_argument('--title', default='DreamerV3 on the Donkey Car simulator')
args = parser.parse_args()


def font(size):
  """A scalable sans font, found through fontconfig; PIL's built-in bitmap font if there is none."""
  found = subprocess.run(['fc-match', '-f', '%{file}', 'sans'], capture_output=True, text=True).stdout
  return ImageFont.truetype(found, size) if found else ImageFont.load_default()


SMALL, MEDIUM, BIG = font(13), font(16), font(30)
rows = [json.loads(line) for line in (args.record / 'steps.jsonl').read_text().splitlines()]
# Keep the first episode only, so the video is one continuous drive.
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
zoom = min((W - 60) / (hi_x - lo_x), (H - 80) / (hi_z - lo_z))


def to_map(x, z):
  px = W / 2 + (x - (lo_x + hi_x) / 2) * zoom
  py = H / 2 + 14 - (z - (lo_z + hi_z) / 2) * zoom
  return px, py


def label(draw, text):
  draw.rectangle([0, 0, W, 22], fill=(0, 0, 0))
  draw.text((8, 3), text, font=SMALL, fill=FG)


def bar(draw, y, name, value, lo, hi, text):
  draw.text((24, y), name, font=MEDIUM, fill=DIM)
  x0, x1 = 130, W - 90
  draw.rectangle([x0, y + 3, x1, y + 17], outline=DIM)
  zero = x0 + (0 - lo) / (hi - lo) * (x1 - x0)
  pos = x0 + (min(max(value, lo), hi) - lo) / (hi - lo) * (x1 - x0)
  draw.rectangle([min(zero, pos), y + 4, max(zero, pos), y + 16], fill=ACCENT)
  draw.text((x1 + 10, y), text, font=MEDIUM, fill=FG)


ffmpeg = subprocess.Popen([
    'ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{2 * W}x{2 * H}',
    '-r', str(HZ), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '27', str(args.out)],
    stdin=subprocess.PIPE)
trail = []
for index, row in enumerate(rows):
  canvas = Image.new('RGB', (2 * W, 2 * H), BG)

  pilot = Image.open(args.record / f"pilot_{row['frame']}.jpg").convert('RGB')
  panel = pilot.resize((W, H), Image.LANCZOS)
  draw = ImageDraw.Draw(panel)
  label(draw, 'PILOT VIEW  (the car\'s only sensor)')
  agent = pilot.resize((64, 64), Image.BILINEAR).resize((112, 112), Image.NEAREST)
  panel.paste(agent, (W - 122, H - 122))
  draw.rectangle([W - 123, H - 123, W - 10, H - 10], outline=FG)
  draw.text((W - 122, H - 140), 'agent input 64x64', font=SMALL, fill=FG)
  canvas.paste(panel, (0, 0))

  panel = Image.open(args.record / f"chase_{row['frame']}.jpg").convert('RGB').resize((W, H), Image.LANCZOS)
  label(ImageDraw.Draw(panel), 'CHASE')
  canvas.paste(panel, (W, 0))

  panel = Image.new('RGB', (W, H), BG)
  draw = ImageDraw.Draw(panel)
  label(draw, 'TOP-DOWN MAP  (track centre line and the car\'s path)')
  if track:
    draw.line([to_map(*p) for p in track + track[:1]], fill=(90, 90, 100), width=9)
  trail.append(to_map(row['x'], row['z']))
  if len(trail) > 1:
    draw.line(trail, fill=TRAIL, width=2)
  x, y = trail[-1]
  draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=ACCENT, outline=FG)
  canvas.paste(panel, (0, H))

  panel = Image.new('RGB', (W, H), BG)
  draw = ImageDraw.Draw(panel)
  label(draw, 'TELEMETRY')
  draw.text((24, 40), args.title, font=MEDIUM, fill=FG)
  draw.text((24, 64), 'policy trained inside a learned world model, camera only', font=SMALL, fill=DIM)
  draw.text((24, 100), f"lap {min(int(row['laps']) + 1, max(1, round(rows[-1]['laps'])))}", font=BIG, fill=FG)
  draw.text((200, 100), f"{index / HZ:5.1f} s", font=BIG, fill=FG)
  draw.text((24, 140), f"{row['laps']:.2f} laps driven", font=MEDIUM, fill=DIM)
  bar(draw, 190, 'steering', row['steer'], -1, 1, f"{row['steer']:+.2f}")
  bar(draw, 225, 'throttle', row['throttle'], 0, 1, f"{row['throttle']:.2f}")
  bar(draw, 260, 'speed', row['speed'], 0, 5, f"{row['speed']:.1f}")
  bar(draw, 295, 'off centre', row['cte'], -3, 3, f"{row['cte']:+.1f}")
  canvas.paste(panel, (W, H))

  ffmpeg.stdin.write(canvas.tobytes())
ffmpeg.stdin.close()
ffmpeg.wait()
print(f'wrote {args.out}: {len(rows)} frames, {len(rows) / HZ:.1f} s, {rows[-1]["laps"]:.2f} laps')
