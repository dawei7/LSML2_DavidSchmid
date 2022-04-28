# Flask
import flask

# General library Imports
import io
import base64
import numpy as np
import cv2
import tensorflow as tf

app = flask.Flask(__name__)

# Load self-trained model from H5-Format
model = tf.keras.models.load_model('face_mask_detection_model.h5')


# Face detection model - pretrained for face detection (3rd party trained model)
cvNet = cv2.dnn.readNetFromCaffe('face_detection/architecture.txt',
                                 'face_detection/weights.caffemodel')


@app.route('/')
def home():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def mask_prediction():
    # Convert uploaded file into cv2 format

    if flask.request.files['image'].filename == '':
        return flask.redirect('/')

    image = flask.request.files.get('image')


    in_memory_file = io.BytesIO()
    image.save(in_memory_file)
    data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    color_image_flag = 1
    success = "Success. Faces detected."
    image = cv2.imdecode(data, color_image_flag)

    img_size = (124, 124)
    assign = {'0': 'No Mask', '1': "Mask"}

    try:
        (h, w) = image.shape[:2]
    except:
        error = "You haven't uploaded a valid image, please try again."
        return flask.render_template("answer.html", error=error)

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    cvNet.setInput(blob)
    detections = cvNet.forward()
    counter = 0
    for i in range(0, detections.shape[2]):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        frame = image[startY:endY, startX:endX]
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5: # Confidence of face detection
            counter+=1
            im = cv2.resize(frame, img_size)
            im = np.array(im) / 255.0
            im = im.reshape(1, 124, 124, 3)
            result = model.predict(im) # Model prediction for face, whether mask or no mask
            if result > 0.5: # Binary classifier, either face with mask or face without mask
                label_Y = 1
            else:
                label_Y = 0
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(image, assign[str(label_Y)], (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255),2)

    if counter == 0:
        success = "Unfortunately there are no Faces detected.\nPlease try again with another image."

    answer = base64.b64encode(cv2.imencode('.png', image)[1]).decode()

    return flask.render_template("answer.html", answer=answer, success=success)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

