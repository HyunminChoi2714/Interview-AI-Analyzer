import speech_recognition as sr
from textblob import TextBlob
from jinja2 import Template
import pdfkit
from datetime import datetime
import os

# --- STEP 1: ANALYSIS LOGIC ---
def analyze_speech(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    # Convert -1 to 1 scale into a 0-100 Confidence Score
    confidence = int((sentiment + 1) * 50)
    
    # Simple keyword check for content
    keywords = ["experience", "python", "team", "project", "led", "solved"]
    found = [word for word in keywords if word in text.lower()]
    
    return {
        "transcript": text,
        "confidence": confidence,
        "tone": "Positive/Confident" if sentiment > 0.1 else "Neutral/Formal",
        "skills": found
    }

# --- STEP 2: REPORT GENERATION ---
def save_report(data):
    html_template = """
    <html>
    <body style="font-family: Arial; padding: 40px;">
        <h1 style="color: #2c3e50;">Interview Performance Report</h1>
        <p>Generated on: {{ date }}</p>
        <hr>
        <h3>Confidence Score: <span style="color: blue;">{{ confidence }}%</span></h3>
        <p><strong>Observed Tone:</strong> {{ tone }}</p>
        <h3>Content Keywords Detected:</h3>
        <ul>
            {% for skill in skills %}<li>{{ skill }}</li>{% endfor %}
        </ul>
        <div style="background: #f9f9f9; padding: 20px; border-left: 5px solid #ccc;">
            <h4>Raw Transcript:</h4>
            <p>"{{ transcript }}"</p>
        </div>
    </body>
    </html>
    """
    
    template = Template(html_template)
    rendered_html = template.render(
        date=datetime.now().strftime("%B %d, %Y"),
        **data
    )

    # Point to your wkhtmltopdf installation
    path_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
    
    # 2. CREATE A CONFIGURATION
    config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)
    
    pdf_name = "Interview_Report.pdf"
    
    # 3. ADD 'configuration=config' TO THE FUNCTION CALL
    pdfkit.from_string(rendered_html, pdf_name, configuration=config)
    
    print(f"\n✅ SUCCESS: Report created as {pdf_name}")

# --- STEP 3: EXECUTION ---
def start_interview():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n[SYSTEM] Calibrating for background noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("[SYSTEM] Microphone is LIVE. Please describe your experience (Speak now):")
        
        audio = recognizer.listen(source)
        print("[SYSTEM] Audio captured. Analyzing...")

    try:
        # Initial transcription
        text = recognizer.recognize_google(audio)
        results = analyze_speech(text)
        save_report(results)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    start_interview()