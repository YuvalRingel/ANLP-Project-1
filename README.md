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
--model             Choose between model variants: "phi-2" for the smaller, lightweight model or "phi-3.5" for the slightly larger, more competitive model. Default is "phi-3.5".
--prompt            Specify the type of prompting method: "zero_shots" for no examples or "few_shots" for examples in the prompt. Default is "few_shots".
--prompt-version    Choose the prompt version: "basic" for simple prompts or "comprehensive" for more detailed ones. Default is "comprehensive".
--output-dir        Specify the directory where the results will be saved. Default is "results/".
--run-all           Use all choices for model, prompt, and prompt-version```
```
Note that you can choose several model, prompts and prompt versions, for example
`python phi.py --model phi-2 phi-3.5 --prompt zero_shots few_shots`
To run all the configurations possible, use the `--run-all` flag:
`python phi.py --run-all`


results are automatically stored in the results folder

## TweetEval
To evaluate TweetEval and visualize the plots, run:
```
python tweet_eval.py
```



## TODO before submitting
- remove the cache variable
