## VIRTUAL TRIAL : MODERNIZING THE WAYS OF E-COMMERCE.







## OBJECTIVE -
With most of the things shifting to virtual mode and competitors trying to get ahead of one, other textile industries took a hit during these tough times. With a Virtual trial Room, Every small textile shops can help their customers to an online trial Room without being afraid of contracting the disease.  The advantage of using this method would be the reduction of time and effort spent in trying out the clothes physically.





## Project Dependencies -
1) OPENCV
2) GLUON-CV
3) NUMPY/MATPLOTLIB
4) WEB-CAM / Microsoft Kinect



## HOW IT WORKS?
With the help of OpenCV and State Of The Art(S.O.T.A) GluonCV We can capture images and project them on humans with some processing to simulate an online trial room.
The actions will include-
1) Capturing Frames Through Web-Cam.
2) Processing Image
3) Projecting generated Image on the client






## WorkFlow
1) Use of OpenCv

``` python
import cv2
img = cv2.imread("test.png", cv2.IMREAD_COLOR)
cv2.imshow("Test Image", img)
## read write show
cv2.waitKey(0)
### Key Used to Destroy The Window
cv2.destroyAllWindows()
```
2) Capturing the video using openCv packages (cv2)

``` python
cv2.videoCapture()
while (true) :
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
```

3) RGB Normalization – OpenCV uses color contrast
based differentiation of objects by detecting the pixels
which reside on the boundaries where colors change
values significantly.
``` python
g_normalized_f.convertTo(g_normalized, CV_8UC1, 255.0);
```

4) S.O.T.A – GluonCV contains a various function which
together helps in detecting the contours of different
objects in a frame.
GluonCV is toolkit for image processing based on the MXNet Framework
``` python
## Intialize the mxnet learning algorithm with a pretrained model
  def load_pretrained_classification_network():
    model = gcv.model_zoo.get_model('MobileNet1.0', pretrained=True, root = M3_MODELS)
    return model

```

5) Augmentation of colours and logos -. Here in our
case, we want the outermost containing contour which
will relate to the T-shirt which the user or test object is
wearing

6) With mxnet deep learning algorithm we impose the
clothes and ornaments to human body
7) It makes the process user interactions with the help
of Numpy/OpenCV packages for edge detection and
Context Embedding.

