from dotenv import load_dotenv,dotenv_values


load_dotenv()
values_env_openai = dotenv_values(".env")

key = values_env_openai['key']
location = values_env_openai['location']
endpoint = values_env_openai['endpoint']
deployment_id_gpt4=values_env_openai['deployment_id_gpt4']  
api_version = values_env_openai['api_version']
MODEL_NAME = values_env_openai['MODEL_NAME']
DIR_PATH = values_env_openai['DIR_PATH']