# model_playground

- To use this app, first clone this repository with _git clone_.

- Then open the base folder and install the following libraries/run the following commands in your preferred python/conda environment.

```
pip install -r requirements.txt
```

- Create a _.env_ in the working folder and populate it as follows:

```
WANDB_API_KEY= your-weightsandbiases-api-key
OPENAI_API_KEY= your-openai-api-key
GROQ_API_KEY= your-groq-api-key
HF_API_KEY= your-huggingface-api-key
TOGETHER_API_KEY = your-togetherai-api-key
```
- Once all that is done, and your environment is ready, run the following command in your working directory, and the streamlit app should be running without any errors.

```
streamlit run app.py
```
