import cv2
import numpy as np
import pyaudio
import scipy.signal
import threading
from flask import Flask, Response, render_template_string
import mediapipe as mp

# Initialize Flask App for streaming
app = Flask(__name__)

# Project Technical Parameters (From CS724 Report)
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 4             # ReSpeaker 4-Mic Array
RATE = 16000             # 16 kHz sampling rate
MIC_SPACING = 0.05       # 5 cm spacing between linear mics
SPEED_OF_SOUND = 343.0   # meters per second

latest_doa_angle = 90.0
video_frame = None
frame_lock = threading.Lock()

# MediaPipe Face Mesh configuration for lip landmark identification
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=5, refine_landmarks=True, min_detection_confidence=0.5)

def calculate_doa(signal_ch1, signal_ch2):
    """Calculates Direction of Arrival (DOA) using Cross-Power Spectrum Phase (Generalized Cross Correlation)"""
    try:
        # Convert to frequency domain via FFT
        fft1 = np.fft.fft(signal_ch1)
        fft2 = np.fft.fft(signal_ch2)
        
        # Calculate Cross Power Spectrum
        cross_power = fft1 * np.conj(fft2)
        normalized_cross_power = cross_power / (np.abs(cross_power) + 1e-12)
        
        # GCC-PHAT via Inverse FFT to get time-delay
        gcc_phat = np.real(np.fft.ifft(normalized_cross_power))
        
        # Find time delay offset (sample shift)
        shift = np.argmax(gcc_phat)
        if shift > CHUNK // 2:
            shift -= CHUNK
            
        time_delay = shift / RATE
        
        # Calculate Angular Position (DOA) via inverse sine
        sin_argument = (time_delay * SPEED_OF_SOUND) / MIC_SPACING
        sin_argument = np.clip(sin_argument, -1.0, 1.0)
        angle_rad = np.arcsin(sin_argument)
        angle_deg = np.degrees(angle_rad) + 90.0 # Shift relative to center spotlight position
        return angle_deg
    except Exception:
        return 90.0

def audio_processing_thread():
    """Handles PyAudio mic stream data collection and background mathematical signal processing"""
    global latest_doa_angle
    p = pyaudio.PyAudio()
    
    try:
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            
            # De-interlace the 4 channels from stream buffer
            ch1 = audio_data[0::CHANNELS]
            ch2 = audio_data[1::CHANNELS]
            
            # Use RMS value to check signal presence (noise reduction filtering rule)
            rms = np.sqrt(np.mean(ch1**2))
            if rms > 500: # Sound threshold limiter
                computed_angle = calculate_doa(ch1, ch2)
                if 0 <= computed_angle <= 180:
                    latest_doa_angle = computed_angle
    except Exception as e:
        print(f"Audio processing engine failed to initialize: {e}")

def get_lip_distance(landmarks):
    """Calculates Euclidean distance between upper and lower lip inner points"""
    # Key landmarks for inner lip opening coordinates
    upper_lip_top = np.array([landmarks[13].x, landmarks[13].y])
    lower_lip_bottom = np.array([landmarks[14].x, landmarks[14].y])
    return np.linalg.norm(upper_lip_top - lower_lip_bottom)

def video_processing_pipeline():
    """Captures camera frames, evaluates human presence & highlights the active target speaker"""
    global video_frame
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
            
        # Flip frame horizontally for natural tracking mirrored feedback
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        active_speaker_box = None
        max_lip_movement = 0.0
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get bounds of current user face
                coords = np.array([[lm.x * w, lm.y * h] for lm in face_landmarks.landmark])
                xmin, ymin = np.min(coords, axis=0).astype(int)
                xmax, ymax = np.max(coords, axis=0).astype(int)
                
                # Check metrics for active lip movement
                lip_dist = get_lip_distance(face_landmarks.landmark)
                
                # Draw standard tracking bounding lines
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                
                if lip_dist > max_lip_movement and lip_dist > 0.015: # Speaking threshold bound
                    max_lip_movement = lip_dist
                    active_speaker_box = (xmin, ymin, xmax, ymax)
            
            # Spotlight: Highlight speaking person dynamically in real time
            if active_speaker_box:
                x1, y1, x2, y2 = active_speaker_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4) # Green highlight box
                cv2.putText(frame, "ACTIVE SPEAKER", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
        # Inject current DOA calculated direction status into screen text
        cv2.putText(frame, f"Estimated Audio DOA: {latest_doa_angle:.2f} deg", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        with frame_lock:
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                video_frame = jpeg.tobytes()

def generate_web_stream():
    """Generates streaming buffer blocks for live Flask interface display"""
    while True:
        with frame_lock:
            if video_frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + video_frame + b'\r\n\r\n')

@app.route('/')
def index():
    # Built-in live monitoring page view template string
    return render_template_string("""
        <html>
            <head><title>CS724 Team 8 Spotlight Monitor</title></head>
            <body style="background-color:#1e1e24; color:white; font-family:sans-serif; text-align:center;">
                <h1>Speaker Spotlight Live Stream Dashboard</h1>
                <img src="/video_feed" width="80%" style="border: 4px solid #3a3a43; border-radius:8px;">
            </body>
        </html>
    """)

@app.route('/video_feed')
def video_feed():
    return Response(generate_web_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Initialize background process threads
    threading.Thread(target=audio_processing_thread, daemon=True).start()
    threading.Thread(target=video_processing_pipeline, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True)
