# How to Batch Image OCR with Generative AI API Calls
<img src="asserts/demo.png" width="1200" height=auto /> </div>
<br>


## Env config
install [uv](https://docs.astral.sh/uv/guides/projects/) and run the following command to install the dependencies:
```
uv sync
```

## Run the demo
Prepare your own OpenAI API key, and run [.env](.env).
- `input_dir`: the directory of your input images, e.g., `sample_images`
- `collection_series`: the collection series you want to use `legacy` (Generative AI Challenge) or `sydow`
- `save-json`: whether to save the OCR results in JSON format
- `save-excel`: whether to save the OCR results in Excel format
```
uv run python -m main --input-dir sample_images --collection-series legacy --save-json --save-excel
```
## Ajust it to your project
You need to create a structured output Pydantic model in [main.py](main.py) and adjust `batch_ocr` function accoding, then change your promptsin [ocr_prompt_default.yml](ocr_prompt_default.yml) to your needs. <br>
You can read more details in [FungariumOCR.pdf](FungariumOCR.pdf). <br>
Enjoy :)