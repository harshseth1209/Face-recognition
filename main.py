import face_recognition
import cv2
import os
import numpy as np

# Function to create face_encoding of all images
path = r"D:\Jupyter\Opencv\Face recognition\sample\\"
list_dir = os.listdir(path)

known_face_encodings = []
known_face_label = []

# Predefined users
def face_encoding(list_dir, known_face_encodings, known_face_label):
    for i in range(len(list_dir)):
        user_name = list_dir[i].split('.')[0]
        known_face_label.append(user_name)
        user_name = face_recognition.load_image_file(path+list_dir[i])
        user_name_new = str(user_name) + '_fe'
        user_name_new = face_recognition.face_encodings(user_name)[0]
        known_face_encodings.append(user_name_new)

face_encoding(list_dir, known_face_encodings, known_face_label) # Function calling



# For new user
paths = "D:\Jupyter\Opencv\Face recognition\sample"

def new_user(name:str, paths):
    new_name = name.split('.')[0]
    known_face_label.append(new_name)
    name = face_recognition.load_image_file(paths+ '\\' + name)
    name_fe = face_recognition.face_encodings(name)[0]
    known_face_encodings.append(name_fe)


# new_user('Husain.jpeg', paths) # Function calling


face_locations = []
face_encodings = []


# To enable Video cam
def camera(face_locations, face_encodings):
    process_this_frame = True
    video = cv2.VideoCapture(0)
    while True:
        ret, frame = video.read()
        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rbg_small_frame = small[:, :, ::-1]
        
        if True:
            face_locations = face_recognition.face_locations(rbg_small_frame)
            face_encodings = face_recognition.face_encodings(rbg_small_frame, face_locations)
            
            face_name = []
            for i in face_encodings:
                match = face_recognition.compare_faces(known_face_encodings, i)
                name = "Unknown"
                
                face_distance = face_recognition.face_distance(known_face_encodings, i)
                best_max_index = np.argmin(face_distance)
                
                if match[best_max_index]:
                    name = known_face_label[best_max_index]
                
                face_name.append(name)
        
        process_this_frame = not process_this_frame
        
        for (top, right, bottom, left), name in zip(face_locations, face_name):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame)
        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()            

print("Web cam is about to enable.")
camera(face_locations, face_encodings)