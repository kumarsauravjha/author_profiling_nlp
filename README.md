# **Age Group Prediction Using Contextual Embeddings**

This project leverages **DistilBERT contextual embeddings** and additional categorical features (gender and occupation) to predict age groups using a neural network. The pipeline involves text preprocessing, feature integration, and model training with early stopping for efficient and robust classification.

---

## **Table of Contents**
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Approach](#approach)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Installation and Usage](#installation-and-usage)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)

---

## **Project Overview**
The goal of this project is to classify individuals into predefined **age groups** based on their textual data, gender, and occupation. The project explores the power of **state-of-the-art embeddings** from DistilBERT combined with additional features to enhance predictive accuracy.

---

## **Dataset**
- **Features Used**:
  - Cleaned text data.
  - Gender (binary encoded).
  - Occupation (target-encoded).  

- **Target**:
  - Multi-class age group classification:
    - Adolescents (13–17)
    - Young Adults (18–30)
    - Adults (31+)

---

## **Approach**
1. **Text Preprocessing**:
   - Removed noise like URLs, HTML tags, and special characters.
   
2. **Embedding Generation**:
   - Used **DistilBERT** to generate 768-dimensional sentence embeddings.

3. **Feature Integration**:
   - Combined contextual embeddings with encoded categorical features (gender and occupation).

4. **Model Training**:
   - Built a Fully Connected Neural Network (FCNN) with early stopping.
   - Used **CrossEntropyLoss** for multi-class classification.

---

## **Model Architecture**
- **Input Features**: 
  - 770 dimensions (768 from DistilBERT + 2 categorical features).  
- **Hidden Layers**:
  - Four fully connected layers with ReLU activations and dropout regularization.  
- **Output Layer**:
  - Fully connected layer for logits corresponding to age group classes.

---

## **Results**
- **Accuracy**: Achieved a test accuracy of **62.59%**, outperforming the random baseline of 33.33%.  
- **Key Insights**:
  - DistilBERT embeddings significantly improved performance.
  - Including gender and occupation enhanced model predictions.  

---

## **Installation and Usage**
### **Prerequisites**
- Python 3.8+
- PyTorch
- Hugging Face Transformers
- Scikit-learn
- tqdm

### **Steps to Run**:
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/age-group-prediction.git
   cd age-group-prediction
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare the dataset and update paths in the script.
4. Run the training pipeline:
   ```bash
   python train.py
   ```

---

## **Future Work**
- **Fine-tuning DistilBERT** to explore its full potential.
- Incorporating additional features like sentiment analysis or topic modeling.
- Experimenting with transformer-based architectures for further improvements.

---

## **Acknowledgments**
- Hugging Face for providing pre-trained models and tokenizers.
- PyTorch for its flexible and powerful deep learning framework.

---

Feel free to customize further with your project details or include visuals, such as charts or sample embeddings! Let me know if you need help refining this further. 🚀
