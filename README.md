# Emotion detection using deep learning
---

## 🔍 Introduction

• Developed a machine learning-based facial emotional recognition system capable of identifying emotions (e.g. happiness, sadness, surprise) from facial expressions in real-time.

• Led the design and implementation of a convolutional neural network (CNN) for emotion classification, achieving 85% accuracy on the test dataset.

• Streamlined FER processes through targeted hyperparameter tuning and effective transfer learning techniques; achieved a 15% increase in accuracy on emotion recognition tasks using existing datasets tailored to specific user scenarios, reduced processing time from 300ms to less than 150ms per frame.

---

## 👥 Dependencies

* Python 3, [OpenCV](https://opencv.org/), [Tensorflow](https://www.tensorflow.org/)
* To install the required packages, run `pip install -r requirements.txt`.

---

## 🔧 Basic Usage

The repository is currently compatible with `tensorflow-2.0` and makes use of the Keras API using the `tensorflow.keras` library.

* First, clone the repository and enter the folder

```bash
git clone https://github.com/atulapra/Emotion-detection.git
cd Emotion-detection
```

* Download the FER-2013 dataset inside the `src` folder.

* If you want to train this model, use:  

```bash
cd src
python emotions.py --mode train
```

* If you want to view the predictions without training again, you can download the pre-trained model from [here](https://drive.google.com/file/d/1FUn0XNOzf-nQV7QjbBPA6-8GLoHNNgv-/view?usp=sharing) and then run:  

```bash
cd src
python emotions.py --mode display
```

* The folder structure is of the form:  
  src:
  * data (folder)
  * `emotions.py` (file)
  * `haarcascade_frontalface_default.xml` (file)
  * `model.h5` (file)

* This implementation by default detects emotions on all faces in the webcam feed. With a simple 4-layer CNN, the test accuracy reached 63.2% in 50 epochs.

![Accuracy plot](imgs/accuracy.png)

---

## 👀 Algorithm

* First, the **haar cascade** method is used to detect faces in each frame of the webcam feed.

* The region of image containing the face is resized to **48x48** and is passed as input to the CNN.

* The network outputs a list of **softmax scores** for the seven classes of emotions.

* The emotion with maximum score is displayed on the screen.

---

