# IMDb Sentiment Classification with Full Finetuning vs LoRA

This project compares **full finetuning** and **parameter-efficient finetuning (LoRA)** using the `distilbert-base-uncased` model on the IMDb moview review dataset.

## Task
Binary sentiment classification of movie reviews into **positive** or **negative**.


## Dataset

- **IMDb**: 50,000 labeled movie reviews (25k train / 25k test)


## Models

| Method            | Trainable Params | Description                                      |
|-------------------|------------------|--------------------------------------------------|
| Full Finetuning   | ~67M (100%)      | All model weights updated                       |
| LoRA (PEFT)       | ~739K (1.1%)     | Only low-rank adapters in attention layers      |

Base model: `distilbert-base-uncased`


## Training Setup

- **Epochs**: 3  
- **Batch Size**: 16  
- **Max Tokens**: 256  
- **Optimizer**: AdamW (lr = 2e-5)  
- **Loss**: CrossEntropyLoss  
- **Evaluation**: Accuracy, Precision, Recall, F1  


## Results

| Strategy         | Test Accuracy | F1 Score | Train Time |
|------------------|---------------|----------|------------|
| Full Finetuning  | 90.62%        | 0.91     | ~1300 min  |
| LoRA             | 89.13%        | 0.89     | ~750 min   |


## Files

| File              | Description                              |
|-------------------|------------------------------------------|
| `main.ipynb`      | Full training notebook                    |
| `logs.txt`        | Raw output of training and evaluation     |
| `requirements.txt`| Python package dependencies               |
| `utils.py`        | Helper functions for data preprocessing   |

