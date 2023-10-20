
# Project Title

This project is a combination of a Streamlit application (`app.py`) and a backend module (`bedrock.py`). 
The Streamlit app provides an interactive interface, while `bedrock.py` handles data retrieval, 
possibly from AWS services like Amazon Kendra.

## Features

- Interactive chat or similar feature via Streamlit
- Data retrieval from various sources
- Advanced CLI options for `bedrock.py`

## Dependencies

To set up the project, install the following Python packages:

```
langchain==0.0.319
boto3>=1.28.27
streamlit
pypdf
chromadb
```

Run the following command to install these packages:

```
pip install -r requirements.txt
```

## Usage

### Running the App via Streamlit

Execute the command:

```
streamlit run app.py bedrock
```

### Debugging and Advanced Usage via CLI

`bedrock.py` can be run individually and supports various command-line switches. 
For example, you can set different personas using an argument parser.

To run `bedrock.py`:

```
python bedrock.py
```

To set a persona:

```
python bedrock.py --persona <PERSONA_NAME>
```

To switch settings within the app, use the `!settings` command.

