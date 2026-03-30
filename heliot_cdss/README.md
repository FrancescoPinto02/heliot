# heliot
Heliot CDSS


## Introduction

To setup the Heliot data pipeline, you must procede as follows:

1. Install Python 3.11, if not already installed.
2. Clone the repository: `git clone https://github.com/FrancescoPinto02/heliot.git`
3. Navigate to the cloned repository directory: `cd /path/to/heliot`
4. Navigate to the pipeline directory: `cd /heliot_cdss`
5. Install poetry: `pip install poetry`
6. Create a new virtual environment with Python 3.11: `poetry env use python3.11`
7. Activate the virtual environment: `poetry env activate`
8. Install app dependencies: `poetry install`
9. Create a `.env ` file in the heliot_cdss folder and set the required environment variables:

   ```
   OPENAI_API_KEY=<your_openai_api_key>
   DATABASE_URL=postgresql://heliot:heliot@localhost:5432/heliot
   REDIS_URL=redis://localhost:6379/0
   HELIOT_API_KEY_PEPPER_V1=<your_api_key_pepper>
   ```
10. Install Docker, if not already installed.
11. Start the required services with Docker Compose: `docker compose up -d`
12. Apply Database migrations using Alembic: `poetry run alembic upgrade head`

### Run the API SERVER
In order to run the Heliot API server, you must run the launch script: `poetry run python -m cdss.heliot.api.main`

### Run the Heliot Web Application
To run the Heliot web Application, simply run: `poetry run streamlit run ./cdss/heliot/app/webapp.py`

### Datasets
In the main folder (`heliot_cdss`) there are the following datasets:
1. patients_synthetic.xslx, the synthetic dataset used for the expertiments
2. real_data_prescriptions.xlsx, the real-world dataset used for the experiments

### Results
In the result folder there are the experimental results, as follows:
1. `gemma` contains the Gemma3 results 
2. `gpt-4o` contains the GPT-4o results
3. `sonnet-4` contains the Claude Sonnet 4.0 results
4. `Usability questionnaire (Answers).csv` contains the answers for the usability study 
