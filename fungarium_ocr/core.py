"""Module providing core functions"""

# pylint: disable=line-too-long, multiple-statements, c-extension-no-member, no-member, no-name-in-module, relative-beyond-top-level, wildcard-import
import os  # interact with system fiels
import glob  # to get file paths
import json  # for JSON output formatting
import base64  # to encode images to send to LLMs
import pandas as pd  # to convert JSON to Excel
from pydantic import BaseModel  # for structured JSON Schema output
from openai import OpenAI  # to use OpenAI models with API keys
from typing import List


class FungariumOCR:
    """Conduct OCR on images from ETH Zurich Fungarium with Generative AI models"""
    def __init__(self, openai_apikey: str = None):
        self.openai_apikey = openai_apikey  # OpenAI API key
        self.client = OpenAI(api_key=self.openai_apikey)

    def get_paths(self, input_dir: str, file_extension: str = '.jpg') -> List[str]:
        """Get paths of all image files with specified extension in the input directory.

        Args:
            input_dir (str): Directory to search for image files.
            file_extension (str, optional): File extension to filter by. Defaults to '.jpg'.

        Returns:
            List[str]: List of paths to matching image files.
        """
        return glob.glob(os.path.join(input_dir, f'*{file_extension}'))

    def visison_model_ocr(self, vsion_model: str = 'gpt-4o', system_prompt: str = None, user_prompt: str = None, image_path: str = None, response_format: BaseModel = None):
        """Perform OCR on an image using a Vision model.

        Args:
            openai_api_key (str): OpenAI API key.
            vsion_model (str): Vision model to use.
            system_prompt (str): System prompt for the model.
            user_prompt (str): User prompt for the model.
            image_path (str): Path to the image file.
            response_format (BaseModel, optional): Pydantic model for structuring the response.

        Returns:
            Any: OCR result from the Vision model, structured according to response_format.
        """

        def encode_image(image_path):
            with open(image_path, 'rb') as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

        base64_image = encode_image(image_path)

        completion = self.client.beta.chat.completions.parse(
            model=vsion_model,
            messages=[
                {
                    'role': 'user',  # I also put sysmtem prompt here as the other way did not work
                    'content': [
                        {'type': 'text',
                         'text': system_prompt + '\n' + user_prompt + '\n' + f'Image name: {os.path.basename(image_path)}'},
                        {'type': 'image_url',
                         'image_url': {
                             'url': f'data:image/jpeg;base64,{base64_image}',
                             'detail': 'auto'}
                         }]
                }],
            response_format=response_format
        )

        result = completion.choices[0].message.parsed  # the OCR result
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

        for image_path in image_paths:
            ocr_result = self.visison_model_ocr(
                image_path=image_path,
                **kwargs
            )
            ocr_results.append(ocr_result.model_dump())

        output_path = os.path.join(input_dir, f'{os.path.basename(input_dir)}.json')

        json_output = json.dumps(ocr_results, indent=2)

        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(json_output)

        try:
            data = json.loads(json_output)
            data_frame = pd.DataFrame(data)
            excel_path = output_path.replace('.json', '.xlsx')
            data_frame.to_excel(excel_path, index=False)
            print(f'Successfully saved Excel file to: {excel_path}')
        except json.JSONDecodeError as e:
            print(f'Error parsing JSON: {e}')
        except Exception as e:
            print(f'Error creating Excel file: {e}')

        return None
