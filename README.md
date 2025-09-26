# Parkinson's Disease Detection Using Speech Analysis

This repository contains the research project for the final year, focusing on the early detection of Parkinson's disease through the analysis of speech signals. The project utilizes machine learning models to identify vocal biomarkers associated with the disease.

## Abstract

Parkinson's disease is a progressive neurodegenerative disorder that affects motor skills, and speech impairment is a common early symptom. This research explores the potential of using speech analysis as a non-invasive, accessible, and cost-effective method for early diagnosis. By leveraging a dataset of voice recordings from both healthy individuals and Parkinson's patients, this project develops and evaluates various machine learning algorithms to build a robust predictive model. The findings, detailed in the accompanying research paper and project report, demonstrate the viability of this approach in distinguishing between healthy and affected individuals.

## 📝 Project Report and Research Paper

This repository includes a comprehensive project report and a research paper that provide in-depth details about the study:

* **[Project-Report-Book.pdf](Project-Report-Book.pdf):** A detailed book-style report covering the entire project, including literature review, methodology, implementation, results, and conclusion.
* **[Project-Research-Paper.pdf](Project-Research-Paper.pdf):** A formal research paper summarizing the project's objectives, methods, and key findings, suitable for publication.

## Dataset

The dataset used in this study consists of voice recordings, from which various speech features have been extracted. These features are crucial for training the machine learning models to detect the subtle vocal impairments caused by Parkinson's disease.

*The specific dataset and its features are detailed in the project report and research paper.*

## 🔬 Methodology

The project follows a structured methodology to ensure a thorough investigation:

1.  **Data Preprocessing:** Cleaning and preparing the dataset for analysis, including handling missing values and normalizing the data.
2.  **Exploratory Data Analysis (EDA):** Analyzing the dataset to uncover patterns, correlations, and key insights that inform the modeling process.
3.  **Feature Selection:** Identifying the most significant vocal features that contribute to the accurate prediction of Parkinson's disease.
4.  **Model Building and Training:** Implementing and training several machine learning models, such as:
    * Support Vector Machine (SVM)
    * Random Forest
    * Gradient Boosting
    * K-Nearest Neighbors (KNN)
5.  **Model Evaluation:** Assessing the performance of the models using various metrics like accuracy, precision, recall, and F1-score to determine the most effective algorithm.
6.  **Result Analysis:** Interpreting the results to draw conclusions about the effectiveness of speech analysis in Parkinson's detection.

## 💻 Technologies and Libraries Used

The entire project is developed in a **Jupyter Notebook (`.ipynb`)**, utilizing the Python programming language and the following key libraries:

* **Scikit-learn:** For implementing the machine learning algorithms.
* **Pandas:** For data manipulation and analysis.
* **NumPy:** For numerical operations.
* **Matplotlib & Seaborn:** For data visualization and creating informative plots.

## How to Run the Project

To replicate the research and run the analysis, follow these steps:

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/Piyush434/parkinson-with-speech-analysis-research-project.git](https://github.com/Piyush434/parkinson-with-speech-analysis-research-project.git)
    ```
2.  **Navigate to the project directory:**
    ```sh
    cd parkinson-with-speech-analysis-research-project
    ```
3.  **Install the required libraries:**
    ```sh
    pip install scikit-learn pandas numpy matplotlib seaborn jupyter
    ```

4.  **Launch Jupyter Notebook:**
    ```sh
    jupyter notebook
    ```
5.  **Open the notebook file:**
    Click on `Major_Project_Exam_Final_Sem_6.ipynb` to open and run the cells.

## Future Scope

This research can be extended in several ways:

* **Real-time Detection:** Developing a mobile application that can record and analyze speech in real-time to provide an instant risk assessment.
* **Larger and Diverse Datasets:** Incorporating more extensive and diverse datasets, including different languages and accents, to improve the model's generalizability.
* **Deep Learning Models:** Exploring advanced deep learning architectures, such as Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs), to potentially capture more complex patterns in speech data.
