import openvino as ov
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from transformers import AutoProcessor, pipeline
import speech_recognition as sr
import time

# --- STEP 1: LOAD MODEL TO INTEL ARC ---
model_id = "openai/whisper-tiny"  # 'tiny' is fastest, 'base' is more accurate
print(f"[PROCESS] Optimizing {model_id} for Intel Arc GPU...")

# This loads the model specifically into your GPU memory
model = OVModelForSpeechSeq2Seq.from_pretrained(
    model_id, 
    export=True, 
    device="GPU", 
    compile=True,
    load_in_8bit=False # Arc handles FP16 better than 8-bit usually
)
processor = AutoProcessor.from_pretrained(model_id)

# Setup the AI pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device="cpu" # The wrapper uses CPU to manage data flow, while OpenVINO runs the math on GPU
)

# --- STEP 2: CAPTURE AUDIO ---
def capture_and_transcribe():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n[READY] Microphone is on. Speak your interview answer...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)

    # Save to a temporary file for the model to read
    with open("temp.wav", "wb") as f:
        f.write(audio.get_wav_data())

    # --- STEP 3: OFFLINE INFERENCE ---
    print("[PROCESS] GPU is transcribing...")
    start_time = time.time()
    
    result = pipe("temp.wav")
    
    end_time = time.time()
    print(f"\n--- OFFLINE REPORT ---")
    print(f"Transcript: {result['text']}")
    print(f"Processing Time: {round(end_time - start_time, 2)} seconds")

if __name__ == "__main__":
    capture_and_transcribe()