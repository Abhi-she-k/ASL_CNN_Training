import multiprocessing as mp

import cv2

import tensorflow as tf


def load_and_preprocess_image(img):
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.image.resize(img, [256, 256])
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)
    return img

def process_frame(q, label):

    model = tf.keras.models.load_model("/Users/abhishek/Desktop/Projects/ASL_CNN_Training/ASL_CNN_Training/Archive Models/ASLCustom.keras")

    while True:
        if(not q.empty()):

            input = q.get()

            predictions = model.predict(input) 
            predicted_class = tf.argmax(predictions[0]).numpy()

            # class_names = ["A", "B", "C", "D", "del", "E", "F", "G", "H", "I", "J", "K", "L", "M",
            #                "N", "nothing", "O", "P", "Q", "R", "S", "space", "T", "U", "V", "W", "X", "Y", "Z"]

            class_names = ["A", "B", "C", "D", "E", "F", "G", "I", "J" , "K" , "L", "M", "N", "NOTHING"]

            print("Predicted class:", class_names[predicted_class])

            print(class_names[predicted_class])
            label.value = class_names[predicted_class]

def frames(q,label):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to read frame.")
            continue

        frame = cv2.flip(frame,1)

        # Convert and crop
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cropped_hand = imgRGB[100:375, 750:1025]

        # Preprocess for model
        processed_img = load_and_preprocess_image(cropped_hand)

        # Add to queue if space
        if not q.full():
            q.put(processed_img)

        label1 = label.value 

        # Draw rectangle on original frame for visualization
        rec = cv2.rectangle(frame, (750, 100), (1025, 375), (255,65,70), 3)
        cv2.putText(rec, label1, (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
        cv2.imshow('ASL Detector', frame)

        # Break loop if 'q' key pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':

    manager = mp.Manager()
    label = manager.Value('s', 'NOTHING') 

    q = mp.Queue(maxsize=10)
    p = mp.Process(target=frames, args=(q,label,))
    p2 = mp.Process(target=process_frame, args=(q,label,))

    p.start()
    p2.start()
    p.join()
    p2.join()