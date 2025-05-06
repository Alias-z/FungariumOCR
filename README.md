# How to Batch Image OCR with Generative AI API Calls
<img src="asserts/demo.png" width="1200" height=auto /> </div>
<br>



## Env config

```
mamba create -n fungi_ocr python=3.10 -y
mamba activate fungi_ocr

mamba install openai pydantic pandas openpyxl jupyter
```

## Run the demo
<a target="_blank" href="https://colab.research.google.com/github/Alias-z/FungariumOCR/blob/llm_course/fungi_ocr.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Prepare your own OpenAI API key, and run [fungi_ocr.ipynb](fungi_ocr.ipynb).

## Ajust it to your project
You only need to change `vsion_model`, `system_prompt`, `user_prompt`, `OutputFormat` according to your needs. <br>
You can read more details in [FungariumOCR.pdf](FungariumOCR.pdf). <br>
Enjoy :)