"""Module providing core functions"""

# pylint: disable=line-too-long, multiple-statements, c-extension-no-member, no-member, no-name-in-module, relative-beyond-top-level, wildcard-import
import os  # interact with system fiels
import glob  # to get file paths
import json  # for JSON output formatting
import yaml  # to load YAML prompts
import base64  # to encode images to send to LLMs
import argparse  # for command line argument parsing
from typing import Literal  # for type hinting
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
    barcode: str = Field(description='Barcode text extracted from the specimen (format: ZT Myc XXXXXXX)')
    division: str = Field(description='Division information that starts the sample information section')
    exicata_number: str = Field(description='Number before the period in the specimen line (e.g., 204)')
    species: str = Field(description='Species name after the period in the specimen line (e.g., Acetabula vulgaris Fuck)')
    matrix_locality: str = Field(description='Location information line extracted as-is (e.g., Ungarn; Comit. Gyor: Bonyretalap)')
    date: str = Field(description='Date information with Roman numeral month and year (e.g., V.1920, X.1924)')
    collector: str = Field(description='Collector name found after "leg." in the specimen information')


class Sydow(BaseModel):
    """Structure for OCR output from fungi specimen images Sydow"""
    barcode: str = Field(description='Barcode text extracted from the specimen (format: ZT Myc XXXXXXX)')
    series_name: str = Field(description='The collection series name, normal on top (e.g. Sydow, Mycotheca germanica)')
    series_number: str = Field(description='The collection series number in front of the taxon (e.g. 103)')
    taxon_name: str = Field(description='Fungi taxon name (e.g., Heterosporium gracile Sacc.)')
    additonal_info: str = Field(description='Additonal info under the Taxon section')
    host: str = Field(description='Fungi host, could be empty, normaly after "Auf" or "Ad" or "In foliis", (e.g. Blättern von Iris germanica.)')
    locality: str = Field(description='Location information line extracted as-is (e.g., Brandenburg: Schlossgarten zu Tamsel.)')
    country: str = Field(description='Country information in English, could be empty, estimated from the other locality (e.g. Germany)')
    collection_date: str = Field(description='Date information in the format of day.month.year (e.g., 16. 7. 1913)')
    collector: str = Field(description='Collector name found after "leg." in the specimen information')


class FungariumOCR:
    """
    Conduct OCR on images from ETH Zurich Fungarium with Generative AI models

    usage:
        uv run python -m main --input-dir sample_images --collection-series legacy --save-json --save-excel
    """
    def __init__(self,
                 openai_apikey: str = os.getenv('OPENAI_API_KEY'),
                 vison_model: str = 'gpt-5-mini'):
        self.openai_apikey = openai_apikey  # OpenAI API key
        self.client = OpenAI(api_key=self.openai_apikey)
        self.vison_model = vison_model  # vision model to use for OCR
        self.default_prompt_path = 'ocr_prompt_default.yml'  # default prompt as the Github demo
        self.sydow_prompt_path = 'ocr_prompt_default_sydow.yml'  # new prompt for Sydow Fungi Exotici Exsiccati and Sydow Mycotheca germanica

    def get_paths(self, input_dir: str, file_extension: str = '.jpg') -> List[str]:
        """Get paths of all image files with specified extension in the input directory.

        Args:
            input_dir (str): Directory to search for image files.
            file_extension (str, optional): File extension to filter by. Defaults to '.jpg'.

        Returns:
            List[str]: List of paths to matching image files.
        """
        return glob.glob(os.path.join(input_dir, f'*{file_extension}'))

    def visison_model_ocr(self, image_path: str = None, ocr_prompt_path: str = None, response_format: BaseModel = None) -> dict:
        """Perform OCR on an image using a Vision model.

        Args:
            image_path (str): Path to the image file.
            ocr_prompt_path (str): Path to the YAML file containing the OCR prompt.
            response_format (BaseModel, optional): Pydantic model for structuring the response.

        Returns:
            Any: OCR result from the Vision model, structured according to response_format.
        """

        def encode_image(image_path):
            with open(image_path, 'rb') as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

        base64_image = encode_image(image_path)
        prompt = load_yaml_prompt(ocr_prompt_path)

        response = self.client.responses.parse(
            model=self.vison_model,
            instructions=prompt['developer'],
            input=[
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'input_text',
                            'text': prompt['user']
                        },
                        {
                            'type': 'input_image',
                            'image_url': f'data:image/jpeg;base64,{base64_image}',
                            'detail': 'auto'
                        }
                    ]
                }
            ],
            text_format=response_format
        )

        result = response.output_parsed  # the OCR result
        return result

    def batch_ocr(self,
                  input_dir: str = None,
                  collection_series: Literal['legacy', 'sydow'] = 'legacy',
                  save_json: bool = True,
                  save_excel: bool = True,
                  **kwargs):
        """Perform OCR on all images in the input directory.

        Args:
            input_dir (str, optional): Directory containing images to process. Defaults to None.
            **kwargs: Additional keyword arguments.
                response_format (BaseModel): Pydantic model for structuring the response.

        Returns:
            str: JSON string containing OCR results.
        """
        if 'sydow' in collection_series:
            prompt_path = self.sydow_prompt_path
            response_format = Sydow
        else:
            prompt_path = self.default_prompt_path
            response_format = Legacy

        image_paths = self.get_paths(input_dir, file_extension='.jpg')  # get all image paths

        ocr_results = []  # to collect OCR result from each image

        for image_path in tqdm(image_paths, total=len(image_paths), desc='Processing images'):
            result = self.visison_model_ocr(
                image_path=image_path,
                ocr_prompt_path=prompt_path,
                response_format=response_format,
                **kwargs
            )
            result = {
                'image_name': os.path.basename(image_path),
                **result.model_dump()}
            ocr_results.append(result)

        if save_json:
            output_path = os.path.join(input_dir, f'{os.path.basename(input_dir)}.json')
            json_output = json.dumps(ocr_results, indent=2)
            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(json_output)
            print(f'Successfully saved JSON file to: {output_path}')

        if save_excel:
            output_path = os.path.join(input_dir, f'{os.path.basename(input_dir)}.xlsx')
            data_frame = pd.DataFrame(ocr_results)
            data_frame.to_excel(output_path, index=False)
            print(f'Successfully saved Excel file to: {output_path}')

        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FungariumOCR: Extract structured text from fungi specimen images')
    parser.add_argument('--input-dir', '-i', type=str, default='sample_images')
    parser.add_argument('--collection-series', '-c', type=str, default='legacy', choices=['legacy', 'sydow'])
    parser.add_argument('--save-json', '-j', action='store_true')
    parser.add_argument('--save-excel', '-e', action='store_true')

    args = parser.parse_args()

    instance = FungariumOCR()
    instance.batch_ocr(
        input_dir=args.input_dir,
        collection_series=args.collection_series,
        save_json=args.save_json,
        save_excel=args.save_excel
    )
    print('OCR processing completed.')
