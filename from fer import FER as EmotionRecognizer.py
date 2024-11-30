from fer import FER as EmotionRecognizer
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import imageio
import matplotlib
import time

matplotlib.use('TkAgg')

# Initialize the emotion detector
emotion_recognizer = EmotionRecognizer(mtcnn=True)

# Start webcam capture
video_capture = cv2.VideoCapture(0)

# Define recording settings
record_rate = 4.3
codec = cv2.VideoWriter_fourcc(*'XVID')
video_output = cv2.VideoWriter('emotion_output.avi', codec, record_rate, (640, 480))

# Set up live bar chart
plt.ion()
chart_figure, chart_axis = plt.subplots()
emotion_categories = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
emotion_bars = chart_axis.bar(emotion_categories, [0]*7, color='lightblue')
plt.ylim(0, 1)
plt.ylabel('Confidence')
plt.title('Live Emotion Detection')
chart_axis.set_xticklabels(emotion_categories, rotation=45)

# GIF writer setup
gif_creator = imageio.get_writer('emotion_chart.gif', mode='I', duration=0.1)

# Collect emotion statistics
emotion_log = []

# Update chart function
def refresh_chart(emotion_data, bars, axis, figure):
    axis.clear()
    axis.bar(emotion_categories, [emotion_data.get(label, 0) for label in emotion_categories], color='lightblue')
    plt.ylim(0, 1)
    plt.ylabel('Confidence')
    plt.title('Live Emotion Detection')
    axis.set_xticklabels(emotion_categories, rotation=45)
    figure.canvas.draw()
    figure.canvas.flush_events()

# Timer start
start_time = time.time()

try:
    while True:
        ret, video_frame = video_capture.read()
        if not ret:
            break

        # Analyze emotions
        detection_result = emotion_recognizer.detect_emotions(video_frame)
        primary_face = None
        largest_area = 0

        # Locate largest face
        for item in detection_result:
            dimensions = item["box"]
            x_coord, y_coord, width, height = dimensions
            face_area = width * height
            if face_area > largest_area:
                largest_area = face_area
                primary_face = item

        if primary_face:
            dimensions = primary_face["box"]
            detected_emotions = primary_face["emotions"]

            emotion_log.append(detected_emotions)

            x_coord, y_coord, width, height = dimensions
            cv2.rectangle(video_frame, (x_coord, y_coord), (x_coord+width, y_coord+height), (0, 255, 0), 2)

            dominant_emotion = max(detected_emotions, key=detected_emotions.get)
            confidence = detected_emotions[dominant_emotion]
            overlay_text = f"{dominant_emotion}: {confidence:.2f}"
            cv2.putText(video_frame, overlay_text, (x_coord, y_coord - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            refresh_chart(detected_emotions, emotion_bars, chart_axis, chart_figure)

            video_output.write(video_frame)

            # Save GIF frames
            chart_figure.canvas.draw()
            gif_frame = np.frombuffer(chart_figure.canvas.tostring_rgb(), dtype='uint8')
            gif_frame = gif_frame.reshape(chart_figure.canvas.get_width_height()[::-1] + (3,))
            gif_creator.append_data(gif_frame)

        cv2.imshow('Live Emotion Analysis', video_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped by user")

finally:
    end_time = time.time()
    print(f"Webcam session duration: {end_time - start_time:.2f} seconds")

    video_capture.release()
    cv2.destroyAllWindows()
    plt.close(chart_figure)
    video_output.release()
    gif_creator.close()

    # Save cumulative data
    data_frame = pd.DataFrame(emotion_log)

    plt.figure(figsize=(10, 10))
    for category in emotion_categories:
        plt.plot(data_frame[category].cumsum(), label=category)
    plt.title('Cumulative Emotion Trends')
    plt.xlabel('Frame')
    plt.ylabel('Cumulative Confidence')
    plt.legend()
    plt.savefig('cumulative_trends.jpg')
    plt.close()
