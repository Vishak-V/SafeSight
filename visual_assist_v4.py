#!/usr/bin/env python

import cv2
import base64
import time
import requests
import json
import io
from gtts import gTTS
import pygame
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- 1. Configuration ---

# Get API key from environment variables
API_KEY = os.getenv("GEMINI_API_KEY")

# This endpoint is for Gemini 1.5 Flash, which supports JSON output mode.
# Using 'generateContent' is the standard method.
API_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={API_KEY}"

# --- MODIFICATION: Revised System Prompt ---
# The system prompt that defines the AI's dual role and output format.
SYSTEM_PROMPT = (
    "You are a time-critical AI Safety Assistant for a person with low vision. "
    "Your sole purpose is to analyze the image for immediate, physical hazards "
    "(e.g., steps, spills, obstacles) and accurately transcribe any visible and "
    "relevant text (signs, labels, warnings). Prioritize safety above all else. "
    "Your entire response MUST be a single, valid JSON object adhering to this exact schema: "
    '{"hazard_detected": "YES/NO", "hazard_warning": "...", "extracted_text": "...", "description": "..."}. '
    "For the 'description' field, provide a concise summary of the scene (e.g., 'Indoor room with a table and chairs' or 'Outside on a street with parked cars'). Use 10-15 words maximum."
    "Do not include markdown backticks (```json) or any text outside the JSON structure."
)

# --- 2. Helper Functions ---

def initialize_systems():
    """Initializes OpenCV webcam and Pygame mixer."""
    print("Initializing systems...")
    
    # Initialize OpenCV Video Capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None, False
    
    # Initialize Pygame Mixer
    try:
        pygame.mixer.init()
    except pygame.error as e:
        print(f"Error initializing pygame.mixer: {e}")
        print("Audio feedback will be disabled. Check your audio drivers.")
        return cap, False # Return cap, but audio_ready is False
    
    print("Systems initialized. Starting real-time loop.")
    print("Press 'q' in the OpenCV window to quit.")
    print("Press 'm' to toggle mute.")
    return cap, True

def process_frame(frame):
    """Encodes a single frame to Base64 JPEG."""
    # STAGE 2: Optimized Encoding
    # Compress frame to JPEG byte buffer
    success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        print("Error: Failed to encode frame.")
        return None
    
    # Convert byte buffer to Base64 string
    base64_image = base64.b64encode(buffer).decode('utf-8')
    return base64_image

def get_analysis_from_api(base64_image):
    """Sends image to API and gets a JSON response."""
    # STAGE 3: LLM API Interaction
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": SYSTEM_PROMPT},
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": base64_image
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            # This forces Gemini to output only JSON
            "responseMimeType": "application/json",
        }
    }
    
    # API Call
    response = requests.post(API_ENDPOINT, headers=headers, json=payload, timeout=10)
    response.raise_for_status() # Raise HTTPError for bad responses

    # STAGE 4: Structured Parsing
    response_data = response.json()
    
    # --- MODIFICATION: Robust JSON parsing ---
    try:
        # With responseMimeType: "application/json", the text part *is* the JSON string
        model_output_json_string = response_data['candidates'][0]['content']['parts'][0]['text']
        return json.loads(model_output_json_string)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
        print(f"Error parsing model's JSON response: {e}")
        print(f"Raw response data: {response_data}")
        # Re-raise the error to be caught by the main loop's try/except block
        raise json.JSONDecodeError("Failed to extract valid JSON from model response", "", 0)


def play_audio_feedback(text, audio_ready, is_muted):
    """
    Generates TTS and plays it synchronously.
    Returns: (bool: continue_running, bool: new_mute_state)
    """
    if not audio_ready:
        print(f"Audio (disabled): {text}")
        # Return original mute state
        return True, is_muted 

    # STAGE 5: Auditory Feedback (Memory-Based TTS)
    try:
        # Stop any currently playing sound immediately.
        pygame.mixer.stop()
        
        # Check if muted
        if is_muted:
            print(f"Audio (muted): \"{text}\"")
            return True, is_muted # Continue running, mute state unchanged

        # Generate TTS audio and write to in-memory buffer
        tts = gTTS(text=text, lang='en')
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0) # Rewind the buffer

        # Synchronized Audio
        sound = pygame.mixer.Sound(mp3_fp)
        sound.play()

        # CRITICAL: Wait for audio to finish
        print(f"Playing audio: \"{text}\"")
        while pygame.mixer.get_busy():
            pygame.time.wait(10) # Yield CPU, check every 10ms
            
            # Listen for 'q' and 'm'
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Interrupted audio with 'q'.")
                pygame.mixer.stop()
                return False, is_muted # Stop loop, mute state unchanged
            elif key == ord('m'):
                is_muted = not is_muted # Toggle mute
                print(f"Audio Muted: {is_muted}")
                if is_muted:
                    pygame.mixer.stop() # Stop playback immediately

    except Exception as e:
        print(f"TTS/Audio Error: {e}")
        # Don't crash, just log and continue
    
    return True, is_muted # Continue running, return current mute state

# --- 3. Main Execution ---

def main():
    cap, audio_ready = initialize_systems()
    if cap is None:
        sys.exit(1)

    is_muted = False

    try:
        while True:
            # Check for keys every loop iteration
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("'q' pressed (main loop), shutting down.")
                break
            elif key == ord('m'):
                is_muted = not is_muted
                print(f"Audio Muted: {is_muted}")
                if is_muted:
                    pygame.mixer.stop() # Stop sound immediately on mute

            # STAGE 1: Capture Frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            # Display the frame (non-blocking)
            cv2.imshow('Real-time Visual Assistance', frame)

            # STAGE 2: Encode Frame
            base64_image = process_frame(frame)
            if base64_image is None:
                continue

            try:
                # STAGES 3 & 4: API Call and Parsing
                analysis = get_analysis_from_api(base64_image)
                
                # --- MODIFICATION: Extract new JSON schema ---
                hazard_detected = analysis.get("hazard_detected", "NO")
                hazard_warning = analysis.get("hazard_warning", "No details provided.")
                extracted_text = analysis.get("extracted_text", "N/A")
                description = analysis.get("description", "No description provided.")

                # --- MODIFICATION: Build feedback string based on priority ---
                feedback_parts = []
                
                # 1. Safety First (Always include hazard status)
                feedback_parts.append(hazard_warning)

                # 2. Text Transcription (If relevant)
                if extracted_text and (extracted_text.lower() != "n/a" or extracted_text.lower() != "none"):
                    feedback_parts.append(f"A sign says: {extracted_text}.")

                # 3. Scene Description
                feedback_parts.append(description)

                # Join all parts into a single string
                feedback_text = " ".join(part for part in feedback_parts if part) # Filter empty

                # STAGE 5: Auditory Feedback
                continue_loop, is_muted = play_audio_feedback(feedback_text, audio_ready, is_muted)
                if not continue_loop:
                    break # User pressed 'q' during audio

            except requests.exceptions.RequestException as e:
                print(f"API Error: {e}")
                # Wait a bit before retrying
                time.sleep(1)
            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                print(f"API Parsing Error: {e}")
                print(f"Received problematic data from API. Skipping frame.")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                break

            # Frame Throttling
            # If a hazard was detected, skip the sleep delay
            if hazard_detected.upper() == "YES":
                print("Hazard detected, skipping delay to process next frame.")
            else:
                time.sleep(0.5)

    finally:
        # Cleanup
        print("Releasing resources...")
        cap.release()
        cv2.destroyAllWindows()
        if audio_ready:
            pygame.mixer.quit()
        print("Shutdown complete.")


if __name__ == "__main__":
    if API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        print("="*50)
        print("ERROR: Please update the 'API_KEY' variable in the script.")
        print("="*50)
    else:
        main()