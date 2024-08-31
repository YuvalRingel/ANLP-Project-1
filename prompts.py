def create_prompt(mode, text):
    if mode == 'zero_shots':
        return _create_zero_shots_prompt(text)
    elif mode == 'few_shots':
        return _create_few_shots_prompt(text)
    else:
        raise ValueError(f"Invalid mode: {mode}")


def _create_zero_shots_prompt(text):
    return f"""Classify the following sentence as Pro-Israeli or Pro-Palestinian. these are the ONLY 2 options.
## Statement:
{text}
## Output:
This statement is """


def _create_few_shots_prompt(text):
    return f"""Classify the following sentence as Pro-Israeli or Pro-Palestinian. these are the ONLY 2 options.
## Statement:
Palestinian people are suffering from the Israeli occupation.
## Output:
This statement is Pro-Palestinian

## Statement:
Israel has the right to defend itself.
## Output:
This statement is Pro-Israeli

## Statement:
{text}
## Output:
This statement is """


def create_exercise_sample_prompt(text, label):
    return f"""## Statement:
{text}

## Output:
This statement is {label}"""




##########
"""
## Statement:
I like peanut butter and jelly sandwiches.
## Output:
This statement is Neutral
"""