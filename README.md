# SafeSight: Low-Latency Visual Alert System (Prototype)

**SafeSight** is a high-speed, minimal-dependency prototype designed as an assistive technology solution. Its core function is to enhance safety for users with low vision by processing a live video feed, identifying potential hazards, and immediately communicating actionable warnings and scene descriptions via auditory feedback.

This prototype is engineered for rapid development and relies entirely on high-level Python libraries and commercial cloud multimodal APIs to achieve fast results without requiring dedicated edge hardware.

-----

## 🚀 Key Features

  * **Real-Time Hazard Detection:** Utilizes cutting-edge multimodal Large Language Models (LLMs) to analyze images for critical physical hazards (e.g., steps, spills, obstacles).
  * **Immediate Auditory Warnings:** Employs a structured prompting strategy to ensure the LLM returns concise, prioritized warnings that are spoken instantly.
  * **High-Quality Text-to-Speech (TTS):** Leverages a cloud-based TTS engine (`gTTS`) combined with memory-based playback (`io.BytesIO`, `pygame`) to ensure high-quality, natural-sounding audio feedback with minimal disk I/O latency.[1, 2]
  * **Synchronous Flow Control:** A simple, sequential pipeline that prioritizes stability and latency management over high frame rate.

## ⚙️ Architecture and Operation

The system operates as a five-stage synchronous pipeline, designed to maintain a critical end-to-end latency target of 1.0 to 2.0 seconds from frame capture to audio delivery.

1.  **Capture & Compression:** The script captures a frame from the local webcam using `opencv-python`.[3] This raw frame (NumPy array) is then compressed into a JPEG byte buffer using `cv2.imencode()`.[4, 5]
2.  **Encoding:** The compressed JPEG bytes are converted into a Base64 string [5], which is the optimal format for inline transmission to vision APIs.[6]
3.  **Cloud Analysis (LLM API):** The Base64 data and a safety-focused system prompt are sent to a high-speed Multimodal LLM (e.g., Gemini Flash or GPT-4V). The prompt forces the model to return a structured JSON output with fields for immediate hazard flags and warnings.[7, 8]
4.  **Parsing & Prioritization:** The script parses the structured JSON response, immediately checking the `hazard_detected` flag.
5.  **Auditory Feedback:** The text (starting with the hazard warning) is passed to `gTTS`, written to a memory buffer, and played back instantly via `pygame`.[1, 2] The loop is blocked until the audio completes, ensuring stability through frame throttling.

## 🛠️ Setup and Dependencies

### Prerequisites

1.  **Python 3.x**
2.  **Webcam** (required for video capture).
3.  **Cloud LLM API Key** (e.g., Google Gemini, OpenAI GPT-4V) and the corresponding API endpoint.

### Installation

Install the required Python libraries using `pip`:

```bash
pip install opencv-python numpy requests gtts pygame
```

### Configuration

Set your API key and endpoint at the beginning of the Python script (or as environment variables, which is recommended).

```python
# API Configuration (PLACEHOLDERS)
API_KEY = "YOUR_LLM_API_KEY_HERE"
API_ENDPOINT = "YOUR_LLM_API_ENDPOINT_HERE"
```

## ▶️ Usage

1.  Ensure your webcam is connected and the dependencies are installed.

2.  Run the script from your terminal:

    ```bash
    python safesight.py
    ```

3.  A window displaying the live video feed will open. The console will show the API response, and the system will speak any detected warnings or scene descriptions.

4.  Press the **'q'** key to close the program and safely release the camera resource.

## ⚠️ Current Limitations (Prototype Stage)

As an overnight, minimal-dependency prototype, **SafeSight** has several functional constraints that should be noted before use:

1.  **Low Frame Rate:** Due to the synchronous API call and frame throttling, the system operates at a low refresh rate (approximately **1–2 FPS**). It is suitable for detecting **static hazards** but cannot reliably track or warn about rapidly moving objects or instantaneous changes in the scene.
2.  **No Contextual Memory:** The system processes each frame in isolation (statelessly). It cannot remember or track objects across sequential frames, which may lead to repetitive descriptions or warnings.
3.  **Cloud Dependency & Cost:** The entire visual analysis relies on a third-party cloud API. This incurs operational costs per frame and requires a persistent, high-speed internet connection to function.
