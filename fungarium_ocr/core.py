"""Module providing core functions"""

# pylint: disable=line-too-long, multiple-statements, c-extension-no-member, no-member, no-name-in-module, relative-beyond-top-level, wildcard-import
import os  # interact with system fiels
import glob  # to get file paths
import json  # for JSON output formatting
import base64  # to encode images to send to LLMs
import cv2  # for image processing
from tqdm import tqdm  # for progress bar
import pandas as pd  # to convert JSON to Excel
from pydantic import BaseModel  # for structured JSON Schema output
from openai import OpenAI  # to use OpenAI models with API keys
from typing import List


image_types = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif', '.ico', '.jfif', '.webp']  # supported image types


class FungariumOCR:
    """Conduct OCR on images from ETH Zurich Fungarium with Generative AI models"""
    def __init__(self, openai_apikey: str = None, api_source: str = 'openai'):
        self.openai_apikey = openai_apikey  # OpenAI API key
        self.api_source = api_source  # API source
        if self.api_source == 'openai':
            self.client = OpenAI(api_key=self.openai_apikey)
        elif self.api_source == 'sph_ethz':
            self.client = OpenAI(api_key=self.openai_apikey, base_url='https://litellm.sph-prod.ethz.ch/')

    def get_paths(self, input_dir: str, file_extension: str = '.jpg') -> List[str]:
        """Get paths of all image files with specified extension in the input directory.

        Args:
            input_dir (str): Directory to search for image files.
            file_extension (str, optional): File extension to filter by. Defaults to '.jpg'.

        Returns:
            List[str]: List of paths to matching image files.
        """
        # Handle case-insensitive matching by creating both lowercase and uppercase patterns
        extension_lower = file_extension.lower()
        extension_upper = file_extension.upper()
        
        # Get files with both lowercase and uppercase extensions
        paths_lower = glob.glob(os.path.join(input_dir, f'*{extension_lower}'))
        paths_upper = glob.glob(os.path.join(input_dir, f'*{extension_upper}'))
        
        # Combine and return unique paths
        return list(set(paths_lower + paths_upper))

    def visison_model_ocr(self, vsion_model: str = 'gpt-4o', system_prompt: str = None, user_prompt: str = None, image_path: str = None, response_format: BaseModel = None, temperature: float = 0.7, resize_ratio: float = 1.0):
        """Perform OCR on an image using a Vision model.

        Args:
            openai_api_key (str): OpenAI API key.
            vsion_model (str): Vision model to use.
            system_prompt (str): System prompt for the model.
            user_prompt (str): User prompt for the model.
            image_path (str): Path to the image file.
            response_format (BaseModel, optional): Pydantic model for structuring the response.
            temperature (float, optional): Controls randomness of the model's output. Defaults to 0.7.
            resize_ratio (float, optional): Ratio to resize the image. 1.0 is original size, 0.5 is half size. Defaults to 1.0.

        Returns:
            Any: OCR result from the Vision model, structured according to response_format.
        """

        def encode_image(image_path):
            # Resize image if resize_ratio is not 1.0
            if resize_ratio != 1.0:
                img = cv2.imread(image_path)
                if img is not None:
                    # Resize based on the provided ratio
                    height, width = img.shape[:2]
                    new_width = int(width * resize_ratio)
                    new_height = int(height * resize_ratio)
                    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    # Encode the resized image
                    _, buffer = cv2.imencode('.jpg', resized_img)
                    return base64.b64encode(buffer).decode('utf-8')
            # Default encoding if no resizing or if image couldn't be loaded
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
            response_format=response_format,
            temperature=temperature
        )

        result = completion.choices[0].message.parsed  # the OCR result
        return result

    def batch_ocr(self, input_dir: str = None, resize_ratio: float = 1.0, **kwargs):
        """Perform OCR on all images in the input directory.

        Args:
            input_dir (str, optional): Directory containing images to process. Defaults to None.
            resize_ratio (float, optional): Ratio to resize images. 1.0 is original size, 0.5 is half size. Defaults to 1.0.
            **kwargs: Additional keyword arguments.
                vsion_model (str): Vision model to use.
                system_prompt (str): System prompt for the model.
                user_prompt (str): User prompt for the model.
                response_format (BaseModel): Pydantic model for structuring the response.
                temperature (float): Controls randomness of the model's output. Defaults to 0.7.

        Returns:
            str: JSON string containing OCR results.
        """
        image_paths = []
        for ext in image_types:
            # Add paths for both lowercase and uppercase extensions
            image_paths.extend(self.get_paths(input_dir, file_extension=ext))

        ocr_results = []  # to collect OCR result from each image

        for image_path in tqdm(image_paths, total=len(image_paths), desc='Processing images'):
            ocr_result = self.visison_model_ocr(
                image_path=image_path,
                resize_ratio=resize_ratio,
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
