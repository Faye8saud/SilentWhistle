import whisper
import spacy
import cv2
import subprocess
import os

# ============ video INPUT ============
video_filename = "videoExample.mp4"  
# ====================================

# قراءة النص باستخدام whisperAi
model = whisper.load_model("medium")
result = model.transcribe(video_filename)
transcription_text = result["text"]

# حفظ النص
with open("transcription.txt", "w", encoding='utf-8') as file:
    file.write(transcription_text)

print("Transcription Done!")

# المعالجة باستخدام spaCy
nlp = spacy.load("en_core_web_sm")
doc = nlp(transcription_text)

# تعريف الايقونات (الرسومات)
icons = {
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

# التحقق من ظهور الكلمات
search_terms = list(icons.keys())
timestamps = []

for segment in result["segments"]:
    text = segment["text"]
    start_time = segment["start"]

    for term in search_terms:
        if term in text:
            timestamps.append((term, start_time))

print("Words & Phrases Found:", timestamps)

# معالجة المقطع باستخدام openCV
cap = cv2.VideoCapture(video_filename)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

temp_video = "temp_output.mp4"
out = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# تحميل وتغيير حجم الايقونات
for key in icons:
    icons[key] = cv2.imread(icons[key], cv2.IMREAD_UNCHANGED)
    if icons[key] is not None:
        icons[key] = cv2.resize(icons[key], (360, 200))

frame_number = 0
icon_display_time = 4  # ثواني

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = frame_number / fps
    for term, timestamp in timestamps:
        if timestamp <= current_time < timestamp + icon_display_time:
            if term in icons and icons[term] is not None:
                icon = icons[term]
                x, y = frame_width - 360, frame_height - 200

                h, w, _ = icon.shape
                for c in range(3):
                    alpha = icon[:, :, 3] / 255.0
                    frame[y:y+h, x:x+w, c] = (
                        icon[:, :, c] * alpha +
                        frame[y:y+h, x:x+w, c] * (1.0 - alpha)
                    )

    out.write(frame)
    frame_number += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing Done, Now merging audio...")

# دمج الصوت مع الفيديو باستخدام FFmpeg
output_filename = "output_video.mp4"
subprocess.run([
    "ffmpeg", "-i", temp_video, "-i", video_filename,
    "-c:v", "copy", "-c:a", "aac", "-strict", "experimental",
    "-map", "0:v:0", "-map", "1:a:0", output_filename, "-y"
])

print(f"✅ Done! Your final video is saved as: {output_filename}")
