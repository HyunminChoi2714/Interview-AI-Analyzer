import openvino as ov
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from transformers import AutoProcessor, pipeline
import speech_recognition as sr
from textblob import TextBlob
from jinja2 import Template
import pdfkit
from datetime import datetime

# --- CONFIGURATION ---
MODEL_ID = "openai/whisper-tiny"
# Update this path to your wkhtmltopdf location
WKHTML_PATH = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'

print(f"[1/3] Loading {MODEL_ID} to Intel Arc GPU...")
model = OVModelForSpeechSeq2Seq.from_pretrained(MODEL_ID, export=True, device="GPU", compile=True)
processor = AutoProcessor.from_pretrained(MODEL_ID)
pipe = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer, 
                feature_extractor=processor.feature_extractor, device="cpu")

def create_report(text):
    print("[3/3] Generating PDF Report...")
    analysis = TextBlob(text)
    confidence = int((analysis.sentiment.polarity + 1) * 50)
    
    html_template = """
    <html><body style="font-family: Arial; padding: 30px;">
        <h1 style="color: #2c3e50;">Interview Analysis (Offline)</h1>
        <hr>
        <h2>Confidence Score: {{ confidence }}%</h2>
        <p><strong>Tone:</strong> {{ tone }}</p>
        <div style="background: #f0f0f0; padding: 15px; border-radius: 5px;">
            <h4>Transcript:</h4><p>"{{ transcript }}"</p>
        </div>
    </body></html>
    """
    
    rendered = Template(html_template).render(
        confidence=confidence,
        tone="Positive" if analysis.sentiment.polarity > 0.1 else "Neutral",
        transcript=text
    )

    config = pdfkit.configuration(wkhtmltopdf=WKHTML_PATH)
    pdfkit.from_string(rendered, "Final_Interview_Report.pdf", configuration=config)
    print("✅ Done! Check 'Final_Interview_Report.pdf'")

def start_session():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n[2/3] Mic is LIVE. Speak your answer...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    
    with open("temp.wav", "wb") as f:
        f.write(audio.get_wav_data())
    
    print("[PROCESS] Transcribing on GPU...")
    result = pipe("temp.wav")
    create_report(result['text'])

if __name__ == "__main__":
    start_session()