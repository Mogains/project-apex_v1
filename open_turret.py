# Press the ▶ Run button in VS Code to open the turret page in Chrome.
import os
import subprocess

page = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_turret.html")
subprocess.run(["open", "-a", "Google Chrome", page])
