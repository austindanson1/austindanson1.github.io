from flask import Flask, request, send_from_directory
import numpy as np
import librosa
import cv2
from moviepy.editor import *

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_video():
    # Access the uploaded file from request.files dictionary
    audio_file = request.files['file']
    
        # Check file extension
    if not audio_file.filename.lower().endswith('.mp3'):
        return "Invalid file format. Please upload an MP3 file.", 400
    
    # You might want to save the audio file locally before processing
    audio_file.save('input.mp3')

    # Load the audio file
    y, sr = librosa.load('input.mp3')

    # Compute the spectral contrast
    S = np.abs(librosa.stft(y))
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)

    # Normalize contrast values to range between 0 and 1
    contrast = (contrast - contrast.min()) / (contrast.max() - contrast.min())

    # Initialize video parameters
    width, height = 1920, 1080  # 1080p
    fps = 30  # frames per second
    seconds = len(y) // sr  # duration of the audio file in seconds

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter('temp_output.mp4', fourcc, fps, (width, height))

    # Create frames
    for i in range(fps * seconds):
        # Create a blank black frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Compute the current radius
        time = i / fps  # current time in seconds
        index = min(int(time * contrast.shape[1] / seconds), contrast.shape[1] - 1)
        radius = int(contrast[0, index] * min(width, height) / 2)

        # Draw the circle
        cv2.circle(frame, (width // 2, height // 2), radius, (255, 255, 255), -1)

        # Write frame to video
        video.write(frame)

    # Release the video file
    video.release()

    # Add the original audio to the video using moviepy
    videoclip = VideoFileClip('temp_output.mp4')
    audioclip = AudioFileClip('input.mp3')
    videoclip = videoclip.set_audio(audioclip)
    videoclip.write_videofile('output.mp4', codec='libx264')

    # Remove the temporary video file
    import os
    os.remove('temp_output.mp4')

    # After processing the video, you might want to send the video file back as response.
    # However, it would be more efficient to store the video file in a cloud storage and send a download URL as response.
    return send_from_directory('.', 'output.mp4')

if __name__ == '__main__':
    app.run()
