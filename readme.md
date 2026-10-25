## Create the two virtual environments

Transformers+PEFT and Unsloth need different package versions, so each gets its own venv:

```
./setup_venvs.sh            # both
```

## generate dataset

You could use either virtual environment. Fill in correct info in .env first.

```
source .venv-hf/bin/activate
python generate_data.py
```

## workflow on transformer+PEFT

```
source .venv-hf/bin/activate
python train-transformer.py --epochs=20
python inference-transformer.py --prompts-file data/eval_prompts.txt --compare-base
```

## workflow on Unsloth

```
source .venv-unsloth/bin/activate
pip install --upgrade unsloth unsloth_zoo
python train-unsloth.py --epochs=20
python inference-unsloth.py --prompts-file data/eval_prompts.txt --compare-base
```

## compare-results

You could use any virtual environment

```
python compare-results.py
```

## upload to AWS S3 bucket

You could use any virtual environment

```
python upload_to_s3.py
```
