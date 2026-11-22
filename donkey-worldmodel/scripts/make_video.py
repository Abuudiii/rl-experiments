"""Compose the demo video from a recorded drive.

  python scripts/make_video.py RECORD_DIR OUT.mp4 [--title TEXT] [--panel 640 480]

RECORD_DIR holds steps.jsonl, track.json and agent_<frame>.jpg from envs/donkey_record.py, plus
pilot_/chase_/overhead_<frame>.png from scripts/replay_views.py. Four panels: pilot camera (with the 64x64
image the agent actually received inset), chase camera, hawkeye camera, and a map of the car's path with
telemetry. Needs ffmpeg on PATH.
"""
import argparse
import json
import pathlib
import subprocess

from PIL import Image, ImageDraw, ImageFont

HZ = 20                  # the simulator sends one frame per 0.05 s of simulated time
BG, FG, DIM = (18, 18, 22), (235, 235, 235), (140, 140, 150)
ACCENT, TRAIL = (80, 220, 120), (240, 200, 60)

parser = argparse.ArgumentParser()
parser.add_argument('record', type=pathlib.Path)
parser.add_argument('out', type=pathlib.Path)
parser.add_argument('--title', default='DreamerV3 on the Donkey Car simulator')
parser.add_argument('--panel', type=int, nargs=2, default=(640, 480), help='width and height of one panel')
parser.add_argument('--lead', type=int, default=0,
                    help='steps the filmed car is ahead of its recorded position (1 for replay_views.py --drive)')
args = parser.parse_args()
W, H = args.panel
S = W / 480              # the layout below is written for 480x360 panels and scaled by this


def px(value):
  return round(value * S)


def font(size):
  """A scalable sans font, found through fontconfig; PIL's built-in bitmap font if there is none."""
  found = subprocess.run(['fc-match', '-f', '%{file}', 'sans'], capture_output=True, text=True).stdout
  return ImageFont.truetype(found, px(size)) if found else ImageFont.load_default()


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
MAP_W, MAP_H = px(210), px(290)   # map area inside the last panel
zoom = min(MAP_W / (hi_x - lo_x), MAP_H / (hi_z - lo_z))


def to_map(x, z):
  return (px(14) + MAP_W / 2 + (x - (lo_x + hi_x) / 2) * zoom,
          px(40) + MAP_H / 2 - (z - (lo_z + hi_z) / 2) * zoom)


def view(name, frame):
  """One camera frame scaled to the panel; None if that view was not filmed."""
  for suffix in ('png', 'jpg'):
    path = args.record / f'{name}_{frame}.{suffix}'
    if path.exists():
      return Image.open(path).convert('RGB').resize((W, H), Image.LANCZOS)
  return None


def label(draw, text):
  draw.rectangle([0, 0, W, px(22)], fill=(0, 0, 0))
  draw.text((px(8), px(3)), text, font=SMALL, fill=FG)


def bar(draw, y, name, value, lo, hi, text):
  x0, x1 = px(250), W - px(62)
  draw.text((x0, px(y)), name, font=SMALL, fill=DIM)
  draw.text((x1 + px(8), px(y + 15)), text, font=SMALL, fill=FG)
  draw.rectangle([x0, px(y + 18), x1, px(y + 30)], outline=DIM)
  zero = x0 + (0 - lo) / (hi - lo) * (x1 - x0)
  pos = x0 + (min(max(value, lo), hi) - lo) / (hi - lo) * (x1 - x0)
  draw.rectangle([min(zero, pos), px(y + 19), max(zero, pos), px(y + 29)], fill=ACCENT)


ffmpeg = subprocess.Popen([
    'ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{2 * W}x{2 * H}',
    '-r', str(HZ), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '22', str(args.out)],
    stdin=subprocess.PIPE)
trail = []
for index, row in enumerate(rows):
  canvas = Image.new('RGB', (2 * W, 2 * H), BG)
  # With --drive the car rolls on about one step before each frame is taken, so the views filmed from the
  # previous step's position show the car where this step's telemetry puts it.
  filmed = rows[max(0, index - args.lead)]['frame']

  panel = view('pilot', filmed)
  draw = ImageDraw.Draw(panel)
  label(draw, 'PILOT VIEW  (the car\'s only sensor, filmed here at higher resolution)')
  # The inset is the frame the simulator sent during the drive, resized the way the agent received it.
  sent = Image.open(args.record / f"agent_{row['frame']}.jpg").convert('RGB')
  side = px(112)
  agent = sent.resize((64, 64), Image.BILINEAR).resize((side, side), Image.NEAREST)
  panel.paste(agent, (W - side - px(10), H - side - px(10)))
  draw.rectangle([W - side - px(11), H - side - px(11), W - px(10), H - px(10)], outline=FG)
  draw.text((W - side - px(10), H - side - px(28)), 'agent input 64x64', font=SMALL, fill=FG)
  canvas.paste(panel, (0, 0))

  panel = view('chase', filmed)
  label(ImageDraw.Draw(panel), 'CHASE')
  canvas.paste(panel, (W, 0))

  panel = view('overhead', filmed)
  if panel:
    label(ImageDraw.Draw(panel), 'HAWKEYE  (camera above the car, looking straight down)')
    canvas.paste(panel, (0, H))

  panel = Image.new('RGB', (W, H), BG)
  draw = ImageDraw.Draw(panel)
  label(draw, 'MAP AND TELEMETRY  ' + args.title)
  if track:
    draw.line([to_map(*p) for p in track + track[:1]], fill=(90, 90, 100), width=px(7))
  trail.append(to_map(row['x'], row['z']))
  if len(trail) > 1:
    draw.line(trail, fill=TRAIL, width=px(2))
  x, y = trail[-1]
  r = px(5)
  draw.ellipse([x - r, y - r, x + r, y + r], fill=ACCENT, outline=FG)
  draw.text((px(250), px(40)), f"lap {min(int(row['laps']) + 1, max(1, round(rows[-1]['laps'])))}", font=BIG, fill=FG)
  draw.text((px(250), px(78)), f"{index / HZ:5.1f} s   {row['laps']:.2f} laps", font=MEDIUM, fill=DIM)
  bar(draw, 125, 'steering', row['steer'], -1, 1, f"{row['steer']:+.2f}")
  bar(draw, 175, 'throttle', row['throttle'], 0, 1, f"{row['throttle']:.2f}")
  bar(draw, 225, 'speed', row['speed'], 0, 5, f"{row['speed']:.1f}")
  bar(draw, 275, 'off centre', row['cte'], -3, 3, f"{row['cte']:+.1f}")
  canvas.paste(panel, (W, H))

  ffmpeg.stdin.write(canvas.tobytes())
ffmpeg.stdin.close()
ffmpeg.wait()
print(f'wrote {args.out}: {len(rows)} frames, {len(rows) / HZ:.1f} s, {rows[-1]["laps"]:.2f} laps')
