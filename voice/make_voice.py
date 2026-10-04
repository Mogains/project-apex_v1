"""
Generate the turret's voice lines with ElevenLabs and save them as MP3s.

Press ▶ Run in VS Code. Only lines that don't have an MP3 yet are generated,
so re-running after adding a line only costs credits for the new one.

Edit LINES below to change what the turret says, then run again.
Writes voice/*.mp3 and voice/lines.js (which camera_turret.html loads).
"""
import hashlib
import json
import os
import re
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"     # "George" — any voice ID from your ElevenLabs voice library
MODEL = "eleven_multilingual_v2"      # best-sounding model; we only pay for each line once

LINES = {
    # said once when the camera starts, and now and then while nothing is in view
    "any": {
        "startup": [
            "Project Apex online. Judges, please keep your hands and feet inside the science fair at all times.",
            "Hello judges. I've been told to be on my best behavior. No promises.",
            "Systems online. Fun fact: I was built in one weekend, powered entirely by snacks.",
            "Apex online. Please rate me ten out of ten. I'll know if you don't.",
        ],
        "idle": [
            "Scanning. Still scanning. This is the most exciting part, I promise.",
            "No targets. Judges, now would be a great time to write down: very impressive.",
            "I'm not saying I deserve first place. But I'm also not not saying that.",
            "If anyone needs me, I'll be over here, menacingly pointing at nothing.",
            "My creator told me to say something smart here. Algorithms. Neural networks. Done.",
            "I was trained on three thousand pictures of one piece of paper. Ask me anything.",
            "Fun fact: I can't actually hurt anyone. I'm basically a very confident desk fan.",
        ],
    },
    "face": {
        "acquired": [
            "Human detected. Act natural.",
            "Oh. Hello there.",
            "I see you. I always see you.",
            "Movement detected. Probably a human. Possibly a very tall raccoon.",
            "Well well well. Look who it is.",
            "Judge detected. Initiating charm mode.",
        ],
        "locked": [
            "Human locked in. Nothing personal.",
            "Locked on. Please hold still, this is for science.",
            "Target acquired. Your face has been added to my list.",
            "Locked in. Don't make this weird.",
            "Gotcha. Smile for the turret.",
            "Locked on. You look like someone who gives full marks.",
        ],
        "lost": [
            "Human lost. Where did you go?",
            "Hello? I was just getting to know you.",
            "Target lost. Hide and seek champion detected.",
            "Gone. Typical human behavior.",
            "Lost you. I'm not crying, you're crying.",
            "Judge lost. Was it something I said?",
        ],
        "firing": [
            "Firing! Sorry, not sorry.",
            "Pew pew.",
            "Deploying maximum annoyance.",
            "Here comes the fun part.",
            "Bombs away. Well, motors away.",
        ],
    },
    "bullseye": {
        "acquired": [
            "Bullseye spotted. It knows what it did.",
            "Ooh, circles. My favorite shape.",
            "Paper target detected. It looks nervous.",
            "I see a bullseye. It sees me. Awkward.",
            "Target found. It was hiding on a piece of paper.",
        ],
        "locked": [
            "Bullseye locked in. Say goodbye, paper.",
            "Locked on. That dot in the middle is getting it.",
            "Target locked. I'm basically a professional now.",
            "Locked in. The circles cannot escape.",
            "Got it. This is my moment.",
            "Locked on. This drawing took four seconds to make, and it is my greatest enemy.",
        ],
        "lost": [
            "Bullseye lost. Coward.",
            "Where did the paper go?",
            "Target lost. The paper has fled the scene.",
            "Lost it. Probably blew away.",
            "It got away. I'll get it next time.",
        ],
        "firing": [
            "Firing! Hold onto your paper.",
            "Pew pew.",
            "Taking the shot.",
            "Aiming for the dot. Hopefully.",
            "Fire in the hole!",
        ],
    },
}


def api_key():
    with open(os.path.join(HERE, "..", "elevenlabs_key.js")) as f:
        return re.search(r'"(sk_[^"]+)"', f.read()).group(1)


def filename(text):
    # stable name per (voice, model, text), so edited lines get a new file and old ones are reused
    h = hashlib.sha1(f"{VOICE_ID}|{MODEL}|{text}".encode()).hexdigest()[:10]
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40]
    return f"{slug}-{h}.mp3"


def generate(text, path, key):
    req = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}?output_format=mp3_44100_128",
        data=json.dumps({"text": text, "model_id": MODEL}).encode(),
        headers={"xi-api-key": key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r, open(path, "wb") as f:
        f.write(r.read())


def main():
    key = api_key()
    manifest, made, chars = {}, 0, 0
    for mode, events in LINES.items():
        for event, texts in events.items():
            entries = []
            for text in texts:
                name = filename(text)
                path = os.path.join(HERE, name)
                if not os.path.exists(path):
                    print("generating:", text)
                    generate(text, path, key)
                    made += 1
                    chars += len(text)
                entries.append({"text": text, "file": f"voice/{name}"})
            manifest.setdefault(mode, {})[event] = entries

    # drop MP3s for lines that were removed
    used = {e["file"].split("/")[-1] for ev in manifest.values() for es in ev.values() for e in es}
    for old in os.listdir(HERE):
        if old.endswith(".mp3") and old not in used:
            os.remove(os.path.join(HERE, old))

    with open(os.path.join(HERE, "lines.js"), "w") as f:
        f.write("// Made by voice/make_voice.py — edit the lines there, not here.\n")
        f.write("window.VOICE_LINES = " + json.dumps(manifest, indent=2) + ";\n")
    print(f"done: {made} new lines generated (~{chars} characters of credits), "
          f"{len(used)} lines ready in voice/")


if __name__ == "__main__":
    main()
