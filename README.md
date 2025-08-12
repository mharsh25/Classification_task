# Classification Task

This project is a classification task to categorize patients into one of six obesity levels. The workflow includes data ingestion, transformation, model training, and prediction, all structured in a modular way with configuration files, exception handling, and logging. A Flask frontend provides a user-friendly interface to interact with the model.

## Project Structure
```
Classification_task/
├── app.py
├── config/
│ ├── configuration.py
│ └── init.py
├── constants.py
├── data_ingestion/
│ ├── data_ingestion.py
│ ├── data_ingestion_config.py
│ └── init.py
├── data_transformation/
│ ├── data_transformation.py
│ ├── data_transformation_config.py
│ └── init.py
├── model_trainer/
│ ├── model_trainer.py
│ ├── model_trainer_config.py
│ └── init.py
├── prediction/
│ ├── prediction.py
│ └── init.py
├── logger.py
├── exception.py
├── utils.py
└── requirements.txt
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
