"""
Build Robo.exe (a double-click launcher for kid_cli.py) and put it on the Desktop.

    python build_robo_exe.py

The .exe is a few KB of C#, compiled with the csc.exe that ships with Windows.
It doesn't bundle Python or the models: it starts Ollama if needed, then runs
kid_cli.py with the Python you build it from. Re-run this script if you move
the project folder or switch Python installs.
"""

import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
OUT = HERE / "launcher"
CSC = Path(r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe")
DESKTOP = Path.home() / "Desktop"


def draw_icon(path: Path) -> None:
    """A friendly robot head, drawn at 256px and saved with the sizes Windows uses."""
    s = 256
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.line((128, 18, 128, 52), fill=(90, 90, 110), width=10)                  # antenna
    d.ellipse((110, 6, 146, 42), fill=(255, 196, 0))                          # antenna light
    d.rounded_rectangle((28, 48, 228, 238), radius=48, fill=(64, 196, 230),
                        outline=(30, 110, 150), width=10)                     # head
    d.ellipse((66, 100, 118, 152), fill="white")                              # eyes
    d.ellipse((138, 100, 190, 152), fill="white")
    d.ellipse((84, 116, 108, 140), fill=(30, 40, 60))
    d.ellipse((156, 116, 180, 140), fill=(30, 40, 60))
    d.arc((78, 140, 178, 212), start=20, end=160, fill=(30, 40, 60), width=12)  # smile
    img.save(path, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])


def main() -> None:
    if not CSC.exists():
        sys.exit(f"C# compiler not found at {CSC}")
    OUT.mkdir(exist_ok=True)
    template = (OUT / "Robo.cs.template").read_text(encoding="utf-8")
    source = template.replace("{{PYTHON}}", sys.executable).replace("{{PROJECT}}", str(HERE))
    (OUT / "Robo.cs").write_text(source, encoding="utf-8")
    draw_icon(OUT / "robo.ico")

    exe = OUT / "Robo.exe"
    result = subprocess.run(
        [str(CSC), "/nologo", "/optimize+", "/target:exe", f"/win32icon:{OUT / 'robo.ico'}",
         f"/out:{exe}", str(OUT / "Robo.cs")],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        sys.exit(f"Compile failed:\n{result.stdout}{result.stderr}")

    shutil.copy2(exe, DESKTOP / "Robo.exe")
    print(f"Built {exe} ({exe.stat().st_size // 1024} KB)")
    print(f"Copied to {DESKTOP / 'Robo.exe'}")
    print(f"It runs: {sys.executable} {HERE / 'kid_cli.py'}")


if __name__ == "__main__":
    main()
