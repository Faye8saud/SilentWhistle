import whisper
import spacy
import numpy as np
import cv2
import arabic_reshaper
from bidi.algorithm import get_display
from moviepy import VideoFileClip
from PIL import Image, ImageDraw, ImageFont
from fuzzywuzzy import fuzz 
import subprocess
import cv2
import numpy as np
import whisper
from fuzzywuzzy import fuzz
from PIL import ImageFont, ImageDraw, Image
from bidi.algorithm import get_display
import arabic_reshaper
import subprocess


icon_map = {
    "هات تريك": "icons/hatrick.png",
    "هاتريك": "icons/hatrick.png",
    "لمسة يد": "icons/touch.png",
    "لمست يد": "icons/touch.png",
    "لمست ": "icons/touch.png",
    "ركنية": "icons/corner.png",
    "الركنية": "icons/corner.png",
    "ركنيه": "icons/corner.png",
    "روكليا": "icons/corner.png",
    "ريمنتادا": "icons/remontada.png",
    "ريمونتادا": "icons/remontada.png",
    "ريمون تادا ": "icons/remontada.png",
    "الريمون تادا": "icons/remontada.png",
    "تسلل": "icons/offside.png",
    "لمسة": "icons/touch.png",
    "التسلل": "icons/offside.png",
    "ضغط عالي": "icons/pressure.png",
    "ضغط": "icons/pressure.png",
}

# تنزيل الأيقونات
loaded_icons = {}
for phrase, path in icon_map.items():
    try:
        loaded_icons[phrase] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    except:
        print(f"Could not load icon: {path}")

#  Overlay Icon Function 
def overlay_icon(frame, icon, x, y, icon_size=(64, 64)):
    icon_resized = cv2.resize(icon, icon_size)

 
    x = min(x, frame.shape[1] - icon_size[0])
    y = min(y, frame.shape[0] - icon_size[1])

    if icon_resized.shape[2] == 4:
        alpha_s = icon_resized[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(3):
            frame[y:y+icon_size[1], x:x+icon_size[0], c] = (
                alpha_s * icon_resized[:, :, c] +
                alpha_l * frame[y:y+icon_size[1], x:x+icon_size[0], c]
            )
    else:
        frame[y:y+icon_size[1], x:x+icon_size[0]] = icon_resized
    return frame

#  Load Whisper Model 
whisper_model = whisper.load_model("medium")

# Transcribe Video 
def transcribe(video_path):
    result = whisper_model.transcribe(video_path, language='ar', fp16=False)
    return result


def normalize_arabic(text):
    if text is None:
        return ""
    return text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا") \
               .replace("ى", "ي").replace("ة", "ه").replace("ئ", "ي") \
               .replace("ؤ", "و")

# Draw Arabic Text

def draw_arabic_text(frame, text, font_path="fonts/Amiri-Regular.ttf", font_size=42, color=(255, 255, 255), position="bottom-right", padding=50):
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)

    font = ImageFont.truetype(font_path, font_size)
    image_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(image_pil)
    text_size = draw.textbbox((0, 0), bidi_text, font=font)
    text_width = text_size[2] - text_size[0]
    text_height = text_size[3] - text_size[1]

    frame_height, frame_width = frame.shape[:2]

    if position == "bottom-right":
        x = frame_width - text_width - padding
        y = frame_height - text_height - padding
    else:
        x, y = padding, padding

    draw.text((x, y), bidi_text, font=font, fill=color)
    return np.array(image_pil)

#  Fuzzy Matching Detection رصد العبارات
KNOWN_PHRASES = {
    "ضغط عالي": "TACTIC",
    "هجمة مرتدة": "TACTIC",
    "انذار": "FOUL",
    "لمست يد": "FOUL",
    "لمسة يد": "FOUL",
    "اللمسة": "FOUL",
    "تسلل": "FOUL",
    "ركنية": "FOUL",
    "ركلة جزاء": "FOUL",
    "ركلة حرة": "FOUL",
    "الريمونتادا": "EVENT",
}

def fuzzy_detect_phrases(segments):
    matches = []
    for segment in segments:
        text = normalize_arabic(segment["text"])
        start_time = segment["start"]

        for known_phrase, label in KNOWN_PHRASES.items():
            if fuzz.partial_ratio(text, known_phrase) >= 80:
                matches.append((known_phrase, start_time))
                break
    return matches

# Main Execution
def main(video_path):
    print("[INFO] Transcribing...")
    result = transcribe(video_path)

    print("[INFO] Detecting simplified phrases (fuzzy matching)...")
    timestamps = fuzzy_detect_phrases(result["segments"])
    print("Detected phrases:", timestamps)

    print("[INFO] Overlaying text on video...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_output = "temp_video.mp4"
    final_output = "output_with_phrases.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    current_phrase = None
    phrase_index = 0
    phrase_start_time = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_count / fps
        if phrase_index < len(timestamps):
            phrase, timestamp = timestamps[phrase_index]
            if current_time >= timestamp:
                current_phrase = phrase
                phrase_start_time = current_time
                phrase_index += 1

        if current_phrase and current_time - phrase_start_time <= 5:
            # frame = draw_arabic_text(frame, current_phrase, font_size=60, position="bottom-right", padding=50)
            # frame = draw_arabic_text(frame, "ملاحظات المعلق:", font_size=36, position="bottom-right", padding=120)

            normalized_phrase = normalize_arabic(current_phrase)
            for key in loaded_icons:
                if key in normalized_phrase:
                    icon_img = loaded_icons[key]
                    if icon_img is not None:
                        print(f"Overlaying icon for: {key}")
                        frame = overlay_icon(frame, icon_img, x=30, y=frame.shape[0] - 100)
                    else:
                        print(f"[WARN] Icon for '{key}' is None")
                    break

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    # التحقق من وجود إطارات في الفيديو المؤقت
    cap_check = cv2.VideoCapture(temp_output)
    has_frames = cap_check.isOpened() and cap_check.read()[0]
    cap_check.release()
    if not has_frames:
        print("[ERROR] temp_video.mp4 has no frames. Skipping audio merge.")
        return

    # دمج الصوت مع الفيديو الأصلي
    print("[INFO] Merging audio from original video...")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", temp_output,
        "-i", video_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        final_output
    ])

    print(f"[DONE] Final output saved to {final_output}")

#  Run Main 
if __name__ == "__main__":
    input_video = "videoExample.mp4"
    main(input_video)
