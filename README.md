# SilentWhistle 🎙⚽️  
An AI-powered tool for making sports commentary videos accessible to the Deaf community by transcribing Arabic audio and overlaying icons that represent key events (e.g., offside, goal, foul).

---

# main Features
- Transcribes Arabic sports commentary using [Whisper AI](https://github.com/openai/whisper)
- Detects sports-related keywords using spaCy NLP
- Overlays intuitive icons (like ⚠️, 🟥, 🎯) on the video using OpenCV
- Supports a range of keywords like:
    - "ضغط عالي" → high pressure
  - "تسلل" → Offside
  - "هاتريك" → Hat-trick
  - "ركنية" → Corner
  - "لمسة يد" → Handball
- Merges video + original audio into a final accessible video file

---

## 🛠️ Technologies Used
- [Whisper AI](https://github.com/openai/whisper) — For Arabic speech transcription
- [spaCy](https://spacy.io/) — For NLP keyword matching
- [OpenCV](https://opencv.org/) — For video and icon overlay
- [FFmpeg](https://ffmpeg.org/) — For audio-video merging
- Python

---

        # Main processing script
▶️ How to Run (Locally)
1. Install dependencies
```bash 
 pip install -r requirements.txt
```
2. Download the Whisper model
```import whisper
model = whisper.load_model("medium")
```
3. Place your input video in the root directory
```bash
python your_script.py
```
5. Your final video will be saved as output_video.mp4
   
📝 To-Do / Coming Soon
✅ Improve timing sync between audio and icon overlays

🧪 Train custom NER for more accurate phrase detection

📺 Live streaming support

🌐 Add multilingual support
