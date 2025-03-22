import os
import numpy as np
import soundfile as sf
import librosa
import sounddevice as sd
import scipy.signal as signal
import time
from datetime import datetime
import threading
import tempfile
import wave

class AudioProcessor:
    """
    Class for recording and processing audio with noise suppression
    """
    def __init__(self, sample_rate=44100, channels=1, chunk_size=1024, 
                 threshold=0.01, silence_timeout=2.0):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.threshold = threshold  # Sound detection threshold
        self.silence_timeout = silence_timeout  # Seconds of silence to stop recording
        
        # Recording state
        self.is_recording = False
        self.audio_buffer = []
        self.recording_thread = None
        self.temp_file = None
        self.last_audio_time = 0
        
        # Callbacks
        self.level_callback = None
        self.status_callback = None
    
    def get_input_devices(self):
        """Get list of available input devices"""
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        return input_devices
    
    def set_callbacks(self, level_callback=None, status_callback=None):
        """Set callbacks for level metering and status updates"""
        self.level_callback = level_callback
        self.status_callback = status_callback
    
    def update_status(self, message):
        """Update status via callback if available"""
        if self.status_callback:
            # Use the callback which should be thread-safe
            self.status_callback(message)
        else:
            print(message)
    
    def update_level(self, level):
        """Update audio level via callback if available"""
        if self.level_callback:
            # Use the callback which should be thread-safe
            self.level_callback(level)
    
    def start_recording(self, device_id=None):
        """Start recording audio from the specified device"""
        if self.is_recording:
            self.update_status("Already recording")
            return
        
        self.update_status("Starting recording...")
        self.audio_buffer = []
        self.last_audio_time = time.time()
        self.is_recording = True
        
        # Create a temporary file for saving audio
        self.temp_file = tempfile.NamedTemporaryFile(
            suffix='.wav', delete=False)
        self.temp_filename = self.temp_file.name
        self.temp_file.close()
        
        # Start recording in a thread
        self.recording_thread = threading.Thread(
            target=self._record_thread, 
            args=(device_id,)
        )
        self.recording_thread.daemon = True
        self.recording_thread.start()
    
    def _record_thread(self, device_id=None):
        """Recording thread function"""
        try:
            # Open a WAV file for writing
            with wave.open(self.temp_filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                
                # Define callback for sounddevice
                def audio_callback(indata, frames, time_info, status):
                    """Callback for sounddevice stream"""
                    if status:
                        self.update_status(f"Status: {status}")
                    
                    # Calculate audio level (RMS)
                    audio_level = np.sqrt(np.mean(indata**2))
                    
                    # Update level through signal-based callback
                    self.update_level(audio_level)
                    
                    # Check if above threshold
                    if audio_level > self.threshold:
                        self.last_audio_time = time.time()
                        
                    # Write audio data to file
                    wf.writeframes((indata * 32767).astype(np.int16).tobytes())
                    
                    # If in idle state for too long, stop recording
                    if self.is_recording and (time.time() - self.last_audio_time) > self.silence_timeout:
                        self.update_status("Silence detected, stopping recording")
                        raise sd.CallbackStop
                
                # Create the stream and handle device selection
                try:
                    stream_args = {
                        'samplerate': self.sample_rate,
                        'channels': self.channels,
                        'callback': audio_callback,
                        'blocksize': self.chunk_size
                    }
                    
                    # Add device selection if specified
                    if device_id is not None:
                        stream_args['device'] = device_id
                    
                    self.update_status("Recording started...")
                    with sd.InputStream(**stream_args):
                        while self.is_recording:
                            sd.sleep(100)
                            
                except Exception as e:
                    self.update_status(f"Stream error: {str(e)}")
                    
        except Exception as e:
            self.update_status(f"Recording error: {str(e)}")
        finally:
            self.is_recording = False
            self.update_status("Recording stopped")
    
    def stop_recording(self):
        """Stop the recording and return the filename"""
        if not self.is_recording:
            self.update_status("Not recording")
            return None
        
        self.update_status("Stopping recording...")
        self.is_recording = False
        
        # Wait for recording thread to finish
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=1.0)
        
        return self.temp_filename
    
    def apply_noise_suppression(self, input_file, output_file=None):
        """
        Apply noise suppression to an audio file
        
        Parameters:
        -----------
        input_file : str
            Path to input audio file
        output_file : str, optional
            Path to save processed audio file, if None generates a new filename
            
        Returns:
        --------
        output_file : str
            Path to the processed audio file
        """
        self.update_status("Applying noise suppression...")
        
        # Generate output filename if not provided
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dirname = os.path.dirname(input_file)
            output_file = os.path.join(dirname, f"processed_{timestamp}.wav")
        
        try:
            # Load audio file
            self.update_status("Loading audio file...")
            y, sr = librosa.load(input_file, sr=None, mono=True)
            
            # Print progress
            self.update_status(f"Audio loaded: {y.shape[0]/sr:.2f} seconds, {sr} Hz")
            
            # 1. Normalize audio
            self.update_status("Normalizing audio...")
            y = librosa.util.normalize(y)
            
            # 2. Estimate noise floor from the first 0.5 seconds (assuming it's silence/background)
            noise_sample = y[:min(int(sr*0.5), len(y))]
            noise_profile = np.mean(np.abs(noise_sample))
            
            # 3. Apply spectral gating for noise reduction
            self.update_status("Applying spectral gating...")
            
            # Compute STFT
            D = librosa.stft(y)
            # Magnitude spectrogram
            magnitude, phase = librosa.magphase(D)
            
            # Estimate noise spectrum
            noise_spectrum = np.mean(np.abs(librosa.stft(noise_sample)), axis=1)
            noise_spectrum = noise_spectrum[:, np.newaxis]
            
            # Apply spectral subtraction with a threshold
            gain_mask = (magnitude - 2 * noise_spectrum)
            gain_mask = np.maximum(gain_mask, 0)
            gain_mask = gain_mask / (magnitude + 1e-10)
            gain_mask = np.minimum(gain_mask, 1)
            
            # Apply mask and reconstruct
            magnitude_filtered = magnitude * gain_mask
            D_filtered = magnitude_filtered * phase
            y_filtered = librosa.istft(D_filtered)
            
            # 4. Apply a slight high-pass filter to remove low rumble
            self.update_status("Applying high-pass filter...")
            nyq = 0.5 * sr
            cutoff = 80.0 / nyq
            b, a = signal.butter(4, cutoff, btype='high')
            y_filtered = signal.filtfilt(b, a, y_filtered)
            
            # 5. Apply a slight low-pass filter to smooth high frequency noise
            cutoff = 8000.0 / nyq
            b, a = signal.butter(4, cutoff, btype='low')
            y_filtered = signal.filtfilt(b, a, y_filtered)
            
            # 6. Final normalization
            y_filtered = librosa.util.normalize(y_filtered)
            
            # Save processed audio
            self.update_status(f"Saving processed audio to {output_file}...")
            sf.write(output_file, y_filtered, sr)
            
            self.update_status("Noise suppression completed")
            return output_file
            
        except Exception as e:
            self.update_status(f"Error during noise suppression: {str(e)}")
            return input_file  # Return original file if processing fails
    
    def extract_features(self, file_path):
        """Extract features from the processed audio file for classification"""
        from batch_process import extract_features as extract_audio_features
        
        self.update_status("Extracting features from audio...")
        features = extract_audio_features(file_path)
        
        if features is None:
            self.update_status("Error: Failed to extract features")
            return None
        
        self.update_status(f"Features extracted: {len(features)} features")
        return features 