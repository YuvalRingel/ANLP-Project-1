## Step 1 - Set Environment

### Create a Virtual Environment
```
python -m venv .venv
```

### Activate 
Then dpeending on your os and environment, activate the virtual environment.<br>
For example, from shell terminal:
```
source .venv/bin/activate.csh
```

### Install Libraries
```
pip install -r requirements.txt
```


## Run scripts
### phi
```
python phi.py {args}
```
use the following args:
```
--model {phi-2,phi-3.5}
--prompt {zero_shots,few_shots} [{zero_shots,few_shots} ...]
```
To finetune, specify `--finetune` if you want to finetune the model before generating responses with it.

results are automatically stored in the results folder

## TweetEval
To evaluate TweetEval and visualize the plots, run:
```
python tweet_eval.py
```



## TODO before submitting
- remove the cache variable
- change the name of the results folder to previosulay generated results or somehting like that any repro won't remove the folders