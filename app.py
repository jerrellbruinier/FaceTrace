import os
import cv2
import numpy as np
import face_recognition
from datetime import datetime
from flask import Flask, render_template, Response
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load training images from subfolders
path = 'Training images'
classNames = []
encodeListKnown = []
allEncodings = []
allClassNames = []

def augment_image(image):
    # Implement basic augmentation: flip the image
    flipped_img = cv2.flip(image, 1)
    return [image, flipped_img]

def preprocess_image(image):
    # Convert BGR (OpenCV default) to RGB (face_recognition requirement)
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Apply histogram equalization
    img_yuv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    processed_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return processed_img

# Traverse subfolders
for person_name in os.listdir(path):
    person_folder = os.path.join(path, person_name)
    if os.path.isdir(person_folder):
        curEncodings = []
        for image_name in os.listdir(person_folder):
            if not image_name.startswith('.'):
                img_path = os.path.join(person_folder, image_name)
                curImg = cv2.imread(img_path)
                if curImg is not None:
                    augmented_images = augment_image(curImg)
                    for img in augmented_images:
                        processed_img = preprocess_image(img)
                        encodings = face_recognition.face_encodings(processed_img)
                        if encodings:
                            curEncodings.append(encodings[0])
        if curEncodings:
            # Add encodings and class names for each image
            for encoding in curEncodings:
                allEncodings.append(encoding)
                allClassNames.append(person_name)

# Split the dataset into 80% training and 20% testing
train_encodings, test_encodings, train_labels, test_labels = train_test_split(
    allEncodings, allClassNames, test_size=0.2, random_state=42)

# Calculate the mean of encodings for each person in the training set
unique_names = list(set(train_labels))
for name in unique_names:
    name_encodings = [train_encodings[i] for i in range(len(train_labels)) if train_labels[i] == name]
    avg_encoding = np.mean(name_encodings, axis=0)
    encodeListKnown.append(avg_encoding)
    classNames.append(name)

print(classNames)
print('Encoding Complete')

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime("%d %B %Y, %H:%M:%S")
            f.writelines(f'\n{name},{dtString}')

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        if not success:
            break
        else:
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)
                threshold = 0.4  # Lower threshold for better unknown face detection

                if faceDis[matchIndex] < threshold:
                    name = classNames[matchIndex].upper()
                else:
                    name = "UNKNOWN"

                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                if name != "UNKNOWN":
                    # Mark attendance immediately when a face is detected
                    markAttendance(name)

            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/attendance_log')
def attendance_log():
    attendance_info = []
    if os.path.exists('Attendance.csv'):
        with open('Attendance.csv', 'r') as f:
            attendance_info = [line.strip().split(',') for line in f.readlines()]
    return render_template('attendance_log.html', attendance_info=attendance_info)

if __name__ == '__main__':
    # Evaluate the model on the test set
    correct_predictions = 0
    for i in range(len(test_encodings)):
        faceDis = face_recognition.face_distance(encodeListKnown, test_encodings[i])
        matchIndex = np.argmin(faceDis)
        threshold = 0.4  # Same threshold used in gen_frames

        if faceDis[matchIndex] < threshold:
            predicted_name = classNames[matchIndex]
        else:
            predicted_name = "UNKNOWN"

        if predicted_name == test_labels[i].upper():
            correct_predictions += 1

    accuracy = correct_predictions / len(test_encodings)
    print(f'Accuracy: {accuracy:.2f}')

    app.run(debug=True)
