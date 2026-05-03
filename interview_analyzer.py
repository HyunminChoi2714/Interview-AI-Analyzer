from textblob import TextBlob
import spacy
import speech_recognition as sr

def record_and_analyze():
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        print(">>> Adjusting for ambient noise... please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print(">>> Microphone is LIVE. Please start the interview (Speak now)...")
        
        # Capture the audio
        audio_data = recognizer.listen(source)
        print(">>> Recording stopped. Processing...")

    try:
        # Convert audio to text using Google's free service (for prototyping)
        # Note: In production, you'd use 'Whisper' for better accuracy.
        transcript = recognizer.recognize_google(audio_data)
        print(f"\n--- Candidate Said ---\n\"{transcript}\"\n")
        
        # Analyze the Tone
        blob = TextBlob(transcript)
        sentiment = blob.sentiment.polarity
        tone = "Confident" if sentiment > 0.1 else "Neutral/Nervous"
        
        print(f"REPORT: Tone is {tone} (Score: {round(sentiment, 2)})")
        
    except sr.UnknownValueError:
        print("Could not understand the audio.")
    except sr.RequestError:
        print("Could not request results from the service.")

if __name__ == "__main__":
    record_and_analyze()

# Load NLP model for content analysis
nlp = spacy.load("en_core_web_sm")

def analyze_interview(transcript, job_keywords):
    print("--- Post-Interview Report ---\n")
    
    # 1. TONE ANALYSIS (Sentiment & Subjectivity)
    blob = TextBlob(transcript)
    sentiment = blob.sentiment.polarity  # -1 (negative) to 1 (positive)
    
    tone_label = "Confident/Positive" if sentiment > 0.2 else "Neutral/Hesitant"
    print(f"Detected Tone: {tone_label} (Score: {round(sentiment, 2)})")

    # 2. CONTENT ANALYSIS (Keyword Matching)
    doc = nlp(transcript.lower())
    found_keywords = [token.text for token in doc if token.text in job_keywords]
    unique_matches = set(found_keywords)
    
    match_percentage = (len(unique_matches) / len(job_keywords)) * 100
    
    print(f"Content Match: {round(match_percentage, 2)}%")
    print(f"Keywords Mentioned: {', '.join(unique_matches) if unique_matches else 'None'}")
    
    # 3. SUMMARY
    print("\nObservation:")
    if match_percentage > 50 and sentiment > 0:
        print("Strong candidate: High technical alignment and positive delivery.")
    else:
        print("Needs Review: Low keyword overlap or neutral delivery.")

# --- MOCK DATA ---
mock_transcript = """
I have extensive experience with Python and SQL. In my last role, 
I managed a cloud database and improved performance by 40%. 
I am very excited about the possibility of leading this team.
"""

required_skills = ["python", "sql", "cloud", "leadership", "javascript"]

# Run the analysis
analyze_interview(mock_transcript, required_skills)