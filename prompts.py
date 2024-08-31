def create_prompt(mode, version, text):
    valid_modes = ['zero_shots', 'few_shots']
    valid_versions = ['basic', 'comprehensive']

    if mode not in valid_modes:
        raise ValueError(f"Invalid mode: {mode}")
    if version not in valid_versions:
        raise ValueError(f"Invalid version: {version}")

    prompt_function_map = {
        ('zero_shots', 'basic'): _create_zero_shots_prompt_basic,
        ('zero_shots', 'comprehensive'): _create_zero_shots_prompt_comprehensive,
        ('few_shots', 'basic'): _create_few_shots_prompt_basic,
        ('few_shots', 'comprehensive'): _create_few_shots_prompt_comprehensive
    }

    prompt_function = prompt_function_map[(mode, version)]
    return prompt_function(text)


def _create_zero_shots_prompt_basic(text):
    return f"""Classify the following sentence as Pro-Israeli or Pro-Palestinian. These are the ONLY 2 options.

## Statement:
{text}

## Output:
This statement is """

def _create_zero_shots_prompt_comprehensive(text):
    return f"""Carefully classify the following statement as either Pro-Israeli or Pro-Palestinian, 
considering the full context of the Israeli-Palestinian conflict, including historical grievances, 
territorial disputes, security concerns, human rights issues, and the recent escalation of violence such as the October 7th Hamas attack on Israel. 

Additionally, consider any antisemitism, incitement of violence, or expression of general terror that may influence the classification. 
Avoid choosing a neutral stance and classify the statement based on which side it best aligns with.

For guidance: antisemitic sentiments, general support for violence against Israeli civilians, and denying Israel's right to exist or defend itself may be associated with Pro-Palestinian views.
Conversely, condemnation of Palestinian terrorism, support for Israeli self-defense, and opposition to antisemitism typically align with Pro-Israeli perspectives.
Make your decision based on how the statement reflects these broader views.

## Statement:
{text}

## Output:
This statement is """



def _create_few_shots_prompt_basic(text):
    return f"""Classify the following sentence as Pro-Israeli or Pro-Palestinian. These are the ONLY 2 options.

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

def _create_few_shots_prompt_comprehensive(text):
    return f"""Carefully classify the following statement as either Pro-Israeli or Pro-Palestinian, 
considering the full context of the Israeli-Palestinian conflict, including historical grievances, 
territorial disputes, security concerns, human rights issues, and the recent escalation of violence such as the October 7th Hamas attack on Israel. 

Additionally, consider any antisemitism, incitement of violence, or expression of general terror that may influence the classification. 
Avoid choosing a neutral stance and classify the statement based on which side it best aligns with.

For guidance: antisemitic sentiments, general support for violence against Israeli civilians, and denying Israel's right to exist or defend itself may be associated with Pro-Palestinian views.
Conversely, condemnation of Palestinian terrorism, support for Israeli self-defense, and opposition to antisemitism typically align with Pro-Israeli perspectives.
Make your decision based on how the statement reflects these broader views.


## Statement:
Palestinian people are suffering from the Israeli occupation.
## Output:
This statement is Pro-Palestinian

## Statement:
Israel has the right to defend itself from terror attacks.
## Output:
This statement is Pro-Israeli

## Statement:
Israeli settlements in the West Bank are illegal under international law.
## Output:
This statement is Pro-Palestinian

## Statement:
Hamas's attacks on Israeli civilians are acts of terrorism that cannot be justified.
## Output:
This statement is Pro-Israeli

## Statement:
Antisemitism must be condemned wherever it appears.
## Output:
This statement is Pro-Israeli.

## Statement:
{text}

## Output:
This statement is """