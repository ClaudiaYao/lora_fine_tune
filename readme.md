## Create the two virtual environments

Transformers+PEFT and Unsloth need different package versions, so each gets its own venv:

```
./setup_venvs.sh            # both
source .venv-hf/bin/activate        # for train-transformer.py / inference-transformer.py
source .venv-unsloth/bin/activate   # for train-unsloth.py / inference-unsloth.py
```

## generate dataset

You could use either virtual environment. Fill in correct info in .env first.

```
python generate_data.py
```

## run train-xx.py

```
source .venv-hf/bin/activate
python train-transformer.py --max-samples=20 --epochs=1
```

```
source .venv-unsloth/bin/activate
python train-unsloth.py --max-samples=400 --epochs=25
```

## run inference-xx.py

```
source .venv-hf/bin/activate
python inference-transformer.py --prompts-file data/eval_prompts.txt --compare-base
```

```
source .venv-unsloth/bin/activate
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
