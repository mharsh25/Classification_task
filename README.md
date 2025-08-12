# Classification Task

This project is a classification task to categorize patients into one of six obesity levels. The workflow includes data ingestion, transformation, model training, and prediction, all structured in a modular way with configuration files, exception handling, and logging. A Flask frontend provides a user-friendly interface to interact with the model.

## Project Structure
```
Classification_task/
├── app.py
├── main.py
├── my_data_report.html
├── requirements.txt
├── setup.py
├── src/
│   ├── data_ingestion.py
│   ├── data_ingestion_config.py
│   ├── data_transformation.py
│   ├── data_transformation_config.py
│   ├── model_trainer.py
│   ├── model_trainer_config.py
│   ├── prediction.py
│   ├── logger.py
│   ├── exception.py
│   └── utils.py
├── templates/
│   └── index.html
└── artifacts/
    └── trained_model.pkl
```


## Features

- Data ingestion and transformation modules for preprocessing.
- Model training module supporting multiple classification models.
- Prediction module integrated with Flask for easy access.
- Exception handling and logging for robustness.
- Modular design using configuration classes for flexibility.

## How to Run

1. Install the dependencies:

```bash
pip install -r requirements.txt
```

2. Start the Flask app:

```bash
python app.py
```

3. Open your browser and navigate to http://localhost:5000 to interact with the model.
