# Project Title

This project is a combination of a Streamlit application (`app.py`) and a backend module (`bedrock.py`). The Streamlit app provides an interactive interface, while `bedrock.py` handles data retrieval, possibly from AWS services like Amazon Kendra.

## Features

- Interactive chat or similar feature via Streamlit
- Data retrieval from various sources
- Utilization of Retrieval-Augmented Generation (RAG) with AWS Kendra for enhanced data handling

## Dependencies

To set up the project, install the following Python packages:

```plaintext
langchain==0.0.319
boto3>=1.28.27
streamlit
pypdf
chromadb
PyYAML
```

Run the following command to install these packages:

```plaintext
pip install -r requirements.txt
```

Rename data/persona.yaml.example to data/persona.yaml.
This file can be modified to customize the AI's persona.

## Usage

### Running the App via Streamlit

Execute the command:

```plaintext
streamlit run app.py bedrock
```

#### Setting a Persona

Personas are defined in the `data/persona.yaml` file. You can update or add new personas by editing this file. The structure to define a persona should follow the existing format within the file.

#### Utilizing RAG with AWS Kendra

This project harnesses the power of Retrieval-Augmented Generation (RAG) in conjunction with AWS Kendra, a highly efficient embedding vector datastore. AWS Kendra facilitates the ingestion of data from a myriad of sources, enhancing the data retrieval process.
