# Football Player Re-Identification

This project provides an end-to-end solution for re-identifying football players in video footage. It includes scripts for training a person re-identification (Re-ID) model and a complete pipeline to process a video, track players, and assign unique IDs to them throughout the match.

The entire project is set up to run seamlessly in Google Colab. just click on (open in colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/flatneuron/football-player-reidentification/blob/main/football_player_reidentification.ipynb)

## Features

-   **End-to-End Pipeline**: From video input to an annotated output video with tracked players and their IDs.
-   **Re-ID Model Training**: Includes code to train your own person re-identification model.
-   **Helper Utilities**: Functions for drawing bounding boxes, and player IDs on video frames.
-   **Colab Ready**: The main notebook `football_player_reidentification.ipynb` is configured to run on Google Colab with just one click. It handles all dependencies and data downloads.

## How to Run

1.  Click on the "Open In Colab" badge above.
2.  Follow the instructions within the `football_player_reidentification.ipynb` notebook. The notebook will guide you through installing dependencies, downloading the necessary data, and running the re-identification pipeline on a sample video.

## Project Structure

```
football-player-reidentification/
├── Train/
│   └── reid_trainer.py   # Contains helper functions for training the Re-ID model.
├── Utils/
│   └── utils.py          # Contains helper functions for the main pipeline and for drawing bounding boxes/IDs.
├── football_player_reidentification.ipynb # The main Google Colab notebook.
└── README.md
```

## Data & Demo

The project uses custom-processed Re-ID data that i creating from the first 5 seconds of the input video for training and includes a demo video showing the final output. This data is hosted on Google Drive and will be downloaded automatically when you run the Colab notebook.

You can also find the data and a demo output video at the following link:
**[Please provide the Google Drive link here]**

The data includes:
- `reid-model-training-data`: Manually cleaned and processed image data for training the person re-identification model.
- A demo output video showing the result of the re-identification pipeline.
