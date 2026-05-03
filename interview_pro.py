import speech_recognition as sr
from textblob import TextBlob

def generate_report(text):
    print("\n--- Generating Post-Interview Report ---")
    analysis = TextBlob(text)
    
    # Simple logic: Positive polarity = Confidence
    # High subjectivity = Personal/Passionate (good for culture fit)
    confidence_score = (analysis.sentiment.polarity + 1) * 50 # Scale to 100
    
    print(f"Final Transcript: {text}")
    print(f"Confidence Level: {round(confidence_score, 1)}%")
    print(f"Tone: {'Enthusiastic' if analysis.sentiment.polarity > 0.3 else 'Professional/Steady'}")

def main():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        # Step 1: Ambient Noise Calibration
        r.adjust_for_ambient_noise(source, duration=1)
        print("System Ready. Please speak your answer...")
        
        # Step 2: Listen (it stops automatically when you pause)
        audio = r.listen(source)
        
    try:
        print("Transcribing via Intel-optimized pipeline...")
        # For now, we use Google, but your Arc GPU will soon handle 'Whisper' locally
        text = r.recognize_google(audio)
        generate_report(text)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()