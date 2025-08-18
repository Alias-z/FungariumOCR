"""Module providing core functions"""

# pylint: disable=line-too-long, multiple-statements, c-extension-no-member, no-member, no-name-in-module, relative-beyond-top-level, wildcard-import
import os  # interact with system fiels
import glob  # to get file paths
import json  # for JSON output formatting
import yaml  # to load YAML prompts
import base64  # to encode images to send to LLMs
from tqdm import tqdm  # for progress bar
from dotenv import load_dotenv  # load environment variables from .env file
import pandas as pd  # to convert JSON to Excel
from pydantic import BaseModel, Field  # for structured JSON Schema output
from openai import OpenAI  # to use OpenAI models with API keys
from typing import List

load_dotenv()  # load environment variables from .env file


def load_yaml_prompt(file_path: str) -> dict:
    """Load a prompt from a YAML file and format for get_llm_completion function

    This function loads YAML prompts and converts them to the format expected by
    get_llm_completion, using 'developer' and 'user' keys instead of 'system' and 'user'.

    Args:
        file_path (str): Path to the YAML file containing the prompt

    Returns:
        dict: The prompt data formatted for get_llm_completion with 'developer' and 'user' keys
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        prompt_data = yaml.safe_load(file)

        for message in prompt_data["messages"]:
            if message["role"] == "developer":
                developer_prompt = message["content"]
            elif message["role"] == "user":
                user_prompt = message["content"]

        prompt_info = {
            "developer": developer_prompt,
            "user": user_prompt
        }
    return prompt_info


class Legacy(BaseModel):
    """Structure for OCR output from fungi specimen images"""
    image_name: str = Field(description='Name or identifier of the processed image')
    barcode: str = Field(description='Barcode text extracted from the specimen (format: ZT Myc XXXXXXX)')
    division: str = Field(description='Division information that starts the sample information section')
    exicata_number: str = Field(description='Number before the period in the specimen line (e.g., 204)')
    species: str = Field(description='Species name after the period in the specimen line (e.g., Acetabula vulgaris Fuck)')
    matrix_locality: str = Field(description='Location information line extracted as-is (e.g., Ungarn; Comit. Gyor: Bonyretalap)')
    date: str = Field(description='Date information with Roman numeral month and year (e.g., V.1920, X.1924)')
    collector: str = Field(description='Collector name found after "leg." in the specimen information')


class SydowFungiExoticiExsiccati(BaseModel):
    """Structure for OCR output from fungi specimen images"""
    image_name: str = Field(description='Name or identifier of the processed image')
    barcode: str = Field(description='Barcode text extracted from the specimen (format: ZT Myc XXXXXXX)')
    division: str = Field(description='Division information that starts the sample information section')
    exicata_number: str = Field(description='Number before the period in the specimen line (e.g., 204)')
    species: str = Field(description='Species name after the period in the specimen line (e.g., Acetabula vulgaris Fuck)')
    matrix_locality: str = Field(description='Location information line extracted as-is (e.g., Ungarn; Comit. Gyor: Bonyretalap)')
    date: str = Field(description='Date information with Roman numeral month and year (e.g., V.1920, X.1924)')
    collector: str = Field(description='Collector name found after "leg." in the specimen information')


class FungariumOCR:
    """
    Conduct OCR on images from ETH Zurich Fungarium with Generative AI models

    usage:
        uv run python -m main
    """
    def __init__(self,
                 openai_apikey: str = os.getenv('OPENAI_API_KEY'),
                 ocr_prompt_path: str = 'ocr_prompt.yml',
                 vison_model: str = 'gpt-5-mini'):
        self.openai_apikey = openai_apikey  # OpenAI API key
        self.client = OpenAI(api_key=self.openai_apikey)
        self.ocr_prompt_path = ocr_prompt_path  # path to the OCR prompt YAML file
        self.vison_model = vison_model  # vision model to use for OCR

    def get_paths(self, input_dir: str, file_extension: str = '.jpg') -> List[str]:
        """Get paths of all image files with specified extension in the input directory.

        Args:
            input_dir (str): Directory to search for image files.
            file_extension (str, optional): File extension to filter by. Defaults to '.jpg'.

        Returns:
            List[str]: List of paths to matching image files.
        """
        return glob.glob(os.path.join(input_dir, f'*{file_extension}'))

    def visison_model_ocr(self, image_path: str = None, response_format: BaseModel = None):
        """Perform OCR on an image using a Vision model.

        Args:
            image_path (str): Path to the image file.
            response_format (BaseModel, optional): Pydantic model for structuring the response.

        Returns:
            Any: OCR result from the Vision model, structured according to response_format.
        """

        def encode_image(image_path):
            with open(image_path, 'rb') as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

        base64_image = encode_image(image_path)
        prompt = load_yaml_prompt(self.ocr_prompt_path)

        image_name = os.path.basename(image_path)

        response = self.client.beta.chat.completions.parse(
            model=self.vison_model,
            messages=[
                {
                    'role': 'developer',
                    'content': prompt['developer']
                },
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': prompt['user']
                        },
                        {
                            'type': 'image_url',
                            'image_url': {
                                'url': f'data:image/jpeg;base64,{base64_image}',
                                'details': 'auto'
                            },
                        }
                    ]
                }
            ],
            response_format=response_format
        )

        result = response.choices[0].message.parsed  # the OCR result

        # response = {
        #     'image_name': image_name,
        #     **response
        # }
        return result

    def batch_ocr(self, input_dir: str = None, **kwargs):
        """Perform OCR on all images in the input directory.

        Args:
            input_dir (str, optional): Directory containing images to process. Defaults to None.
            **kwargs: Additional keyword arguments.
                vsion_model (str): Vision model to use.
                system_prompt (str): System prompt for the model.
                user_prompt (str): User prompt for the model.
                response_format (BaseModel): Pydantic model for structuring the response.

        Returns:
            str: JSON string containing OCR results.
        """
        image_paths = self.get_paths(input_dir, file_extension='.jpg')  # get all image paths

        ocr_results = []  # to collect OCR result from each image

        for image_path in tqdm(image_paths, total=len(image_paths), desc='Processing images'):
            ocr_result = self.visison_model_ocr(
                image_path=image_path,
                **kwargs
            )
            print(ocr_result)
            ocr_results.append(ocr_result.model_dump())

        output_path = os.path.join(input_dir, f'{os.path.basename(input_dir)}.json')

        # json_output = json.dumps(ocr_results, indent=2)
        

        # with open(output_path, 'w', encoding='utf-8') as file:
        #     file.write(json_output)

        # try:
        #     data = json.loads(json_output)
        #     data_frame = pd.DataFrame(data)
        #     excel_path = output_path.replace('.json', '.xlsx')
        #     data_frame.to_excel(excel_path, index=False)
        #     print(f'Successfully saved Excel file to: {excel_path}')
        # except json.JSONDecodeError as e:
        #     print(f'Error parsing JSON: {e}')
        # except Exception as e:
        #     print(f'Error creating Excel file: {e}')

        return None


if __name__ == '__main__':
    instance = FungariumOCR()
    _ = instance.batch_ocr(input_dir='sample_images', response_format=Legacy)
    print('OCR processing completed.')
