🐶🐱 Dog vs Cat Image Classifier

This project is a simple deep learning model built using TensorFlow to classify images as either Dog or Cat. It includes a Jupyter Notebook for training the model and a Flask web app for real-time image prediction.

---

📁 Project Structure

```
animal_classifier/
├── train/                  # Training images
│   ├── cats/               # Cat images
│   └── dogs/               # Dog images
├── model.h5                # Trained model
├── main.py                 # Training script
├── app.py                  # Flask web app
├── Dog_vs_Cat_Classifier_Notebook.ipynb
├── Dog_vs_Cat_Classifier_Project_Report.pdf
└── README.md               # Project instructions
```

---

🚀 How to Run the Project

1. Clone the Repo

```bash
git clone https://github.com/YourUsername/animal_classifier.git
cd animal_classifier
```

2. Install Dependencies

```bash
pip install tensorflow matplotlib numpy flask
```

3. Train the Model (if not already trained)

```bash
python main.py
```

4. Run Flask App

```bash
python app.py
```

Then open your browser and go to:  
`http://127.0.0.1:5000`

Upload an image and get the prediction!

🧠 Model Architecture

- Conv2D → ReLU → MaxPooling
- Conv2D → ReLU → MaxPooling
- Conv2D → ReLU → MaxPooling
- Flatten → Dense(64) → Dense(1 with Sigmoid)


📦 Dependencies

- Python 3.x
- TensorFlow
- NumPy
- Flask
- Matplotlib


📌 Notes

- The model is trained on a small dataset of 20 images (10 dogs & 10 cats).
- For better accuracy, increase the dataset size and training epochs.
- The `.h5` model file is saved and reused by Flask app.


📄 Report & Notebook

- 📘 `Dog_vs_Cat_Classifier_Notebook.ipynb`: Training code with explanations  
- 📄 `Dog_vs_Cat_Classifier_Project_Report.pdf`: Project overview in PDF


✅ Author

Jawaharsrinath  
July 2025
