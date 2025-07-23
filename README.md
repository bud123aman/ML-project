# Face Recognition System using Siamese Networks

This project implements a real-time **Face Recognition System** leveraging the power of **Siamese Neural Networks** with a custom L1 distance layer. The system is designed to verify identities by comparing a live input image against a set of known verification images.

---

## Project Overview

This repository contains the code for building and deploying a face recognition system using a **Siamese Network architecture**. Unlike traditional classification models that categorize an input into predefined classes, Siamese networks learn a similarity function. This allows the model to determine if two inputs (in this case, faces) belong to the same person or different people.

The project covers:
* **Data Collection**: Setting up directories and capturing custom "anchor" and "positive" images via webcam, alongside using external "negative" (LFW) dataset images.
* **Data Preprocessing**: Transforming raw images into a suitable format for the neural network.
* **Siamese Network Architecture**: Building a custom convolutional neural network (CNN) for embedding generation and integrating a custom L1 Distance layer to calculate similarity between face embeddings.
* **Model Training**: Training the Siamese model using a contrastive loss function to learn effective facial embeddings.
* **Real-time Verification**: Implementing a live webcam feed to capture input images and perform real-time identity verification against a set of known faces.

---

## Key Features

* **Custom Data Collection**: Directly capture your own `anchor` and `positive` images using your webcam for a personalized dataset.
* **Siamese Network Implementation**: A robust CNN-based embedding model coupled with a custom L1 Distance layer for similarity computation.
* **Efficient Data Pipeline**: Utilizes `tf.data` for optimized loading, caching, and shuffling of image pairs.
* **Live Face Verification**: Integrates with OpenCV to perform real-time identity checks against a gallery of known faces.
* **TensorFlow/Keras**: Built entirely with TensorFlow's Keras API for ease of development and scalability.

---

## Technologies Used

* **Python**: The core programming language.
* **TensorFlow/Keras**: For building, training, and evaluating the deep learning models.
* **OpenCV (`cv2`)**: For webcam access and real-time image capture.
* **NumPy**: For numerical operations.
* **Matplotlib**: For visualizing images and model insights.
* **`uuid`**: For generating unique filenames during data collection.
* **LFW Dataset**: Used as a source for `negative` (impostor) images.

---

## Installation

To get this project up and running on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd face-recognition-siamese-network
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```
3.  **Install the required libraries:**
    ```bash
    pip install tensorflow opencv-python numpy matplotlib
    ```
4.  **Create necessary data directories:**
    ```bash
    mkdir -p data/positive data/negative data/anchor
    mkdir -p application_data/input_image application_data/verification_images
    ```
    *(The `mkdir -p` command creates parent directories if they don't exist and doesn't throw an error if the directory already exists.)*

5.  **Download and Prepare the LFW (Labeled Faces in the Wild) dataset:**
    * Download the **"All images as gzipped tar file"** from the official LFW website: [http://vis-www.cs.umass.edu/lfw/](http://vis-www.cs.umass.edu/lfw/)


---


## Model Architecture Details

The core of this system is a **Siamese Neural Network**. It operates by learning a similarity metric between pairs of inputs.

The network comprises:

1.  **Embedding Model (`make_embedding` function):** This is a Convolutional Neural Network (CNN) responsible for converting an input facial image (100x100x3 pixels) into a dense, lower-dimensional vector representation (an "embedding" of size 4096). This embedding aims to capture the unique features of a face.

    ```python
    def make_embedding():
        inp = Input(shape=(100,100,3), name='input_image')
        # First Block
        c1 = Conv2D(64, (10,10), activation='relu')(inp)
        m1 = MaxPooling2D(64, (2,2), padding='same')(c1)
        # Second Block
        c2 = Conv2D(128, (7,7), activation='relu')(m1)
        m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
        # Third Block
        c3 = Conv2D(128, (4,4), activation='relu')(m2)
        m3 = MaxPooling2D(64, (2,2), padding='same')(c3)
        # Final Embedding Block
        c4 = Conv2D(256, (4,4), activation='relu')(m3)
        f1 = Flatten()(c4)
        d1 = Dense(4096, activation='sigmoid')(f1)
        return Model(inputs=[inp], outputs=d1, name='embedding')
    ```

    **Embedding Model Summary:**
    ```
    Model: "embedding"
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
    ┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
    │ input_image (InputLayer)        │ (None, 100, 100, 3)    │             0 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ conv2d (Conv2D)                 │ (None, 91, 91, 64)     │        19,264 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ max_pooling2d (MaxPooling2D)    │ (None, 46, 46, 64)     │             0 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ conv2d_1 (Conv2D)               │ (None, 40, 40, 128)    │       401,536 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ max_pooling2d_1 (MaxPooling2D)  │ (None, 20, 20, 128)    │             0 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ conv2d_2 (Conv2D)               │ (None, 17, 17, 128)    │       262,272 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ max_pooling2d_2 (MaxPooling2D)  │ (None, 9, 9, 128)      │             0 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ conv2d_3 (Conv2D)               │ (None, 6, 6, 256)      │       524,544 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ flatten (Flatten)               │ (None, 9216)           │             0 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ dense (Dense)                   │ (None, 4096)           │    37,752,832 │
    └─────────────────────────────────┴────────────────────────┴───────────────┘
     Total params: 38,960,448 (148.62 MB)
     Trainable params: 38,960,448 (148.62 MB)
     Non-trainable params: 0 (0.00 B)
    ```

2.  **L1 Distance Layer (`L1Dist` class):** This is a custom Keras layer that computes the absolute difference (L1 distance) element-wise between the two embeddings generated by the Siamese branches. This difference vector highlights where the embeddings diverge.

    ```python
    class L1Dist(Layer):
        def __init__(self, **kwargs):
            super().__init__()
        def call(self, input_embedding, validation_embedding):
            return tf.math.abs(input_embedding - validation_embedding)
    ```

3.  **Siamese Model (`make_siamese_model` function):** This top-level model takes two input images (an anchor and a validation image), passes each through the shared `embedding` model, calculates their L1 distance using the `L1Dist` layer, and then feeds this distance vector into a final `Dense` layer with a sigmoid activation. The output is a probability score between 0 and 1, indicating similarity. A score closer to 1 means the faces are likely the same person, while a score closer to 0 means they are different.

    ```python
    def make_siamese_model():
        input_image = Input(name='input_img', shape=(100,100,3))
        validation_image = Input(name='validation_img', shape=(100,100,3))

        siamese_layer = L1Dist()
        siamese_layer._name = 'distance' # Renaming for clarity in summary
        distances = siamese_layer(embedding(input_image), embedding(validation_image))

        classifier = Dense(1, activation='sigmoid')(distances)
        return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')
    ```

    **Siamese Model Summary:**
    ```
    Model: "SiameseNetwork"
    ┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
    ┃ Layer (type)        ┃ Output Shape      ┃    Param # ┃ Connected to      ┃
    ┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
    │ input_img           │ (None, 100, 100,  │          0 │ -                 │
    │ (InputLayer)        │ 3)                │            │                   │
    ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
    │ validation_img      │ (None, 100, 100,  │          0 │ -                 │
    │ (InputLayer)        │ 3)                │            │                   │
    ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
    │ embedding           │ (None, 4096)      │ 38,960,448 │ input_img[0][0],  │
    │ (Functional)        │                   │            │ validation_img[0… │
    ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
    │ l1_dist_4 (L1Dist)  │ (None, 4096)      │          0 │ embedding[0][0],  │
    │                     │                   │            │ embedding[1][0]   │
    ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
    │ dense_5 (Dense)     │ (None, 1)         │      4,097 │ l1_dist_4[0][0]   │
    └─────────────────────┴───────────────────┴────────────┴───────────────────┘
     Total params: 38,964,545 (148.64 MB)
     Trainable params: 38,964,545 (148.64 MB)
     Non-trainable params: 0 (0.00 B)
    ```

---

## Model Evaluation

During training, the model uses `BinaryCrossentropy` as the loss function and is optimized with Adam (learning rate 1e-4). After training, `Precision` and `Recall` metrics can be used to assess performance. The `verify` function uses two thresholds:

* **Detection Threshold (e.g., 0.9):** Individual prediction score (from 0 to 1) above which a single comparison is considered a "match."
* **Verification Threshold (e.g., 0.7):** The proportion of positive matches (from the detection threshold) out of all verification images. If this proportion exceeds the verification threshold, the identity is confirmed.

---

## Contributing

I welcome contributions to this project! Whether it's reporting a bug, suggesting a new feature, improving documentation, or submitting code, your help is highly appreciated.


---



## Contact

For any questions or collaborations, please reach out at amansinghbudhala15@gmail.com

---
