import openai
from tqdm import tqdm

def setup_api_key(api_key_file="secret.key"):
    openai.api_key_path = api_key_file

def run_gpt(prompt, model="davinci", temperature=0):
    response = None

    if model == "gpt-3.5-turbo":
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        response = response["choices"][0]["message"]["content"]
    else:
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            temperature=temperature
        )
        response = response["choices"][0]["text"]

    return response

def generate_prompt(text, target_delta, feature="speed"):
    prompt = ""

    if feature == "speed":
        prompt += (
            "Speed is a measure of how quickly content moves in a given text and is "
            "calculated as the distance traveled by consecutive windows of text. More specifically, "
            "we break the text into three word chunks, compute the word embeddings of every chunk, and "
            "compute speed as the average distance between consecutive chunks.\n\n"
        )
        
        # TODO: change prompt/examples
        prompt += ()


    elif feature == "volume":
        prompt += (
            "Volume captures the amount of information covered in a piece of text. We break "
            "the text into three word chunks, compute the word embeddings of every chunk, and compute "
            "volume as the size of the minimum volume ellipsoid that contains all chunk embeddings.\n\n"
        )
        
        # TODO: change prompt/examples
        prompt += ()

    elif feature == "circuitousness":
        prompt += (
            "Circuitousness measures how indirectly content is covered. We break the text "
            "into three word chunks, compute the word embeddings of every chunk, and compute "
            "circuitousness as the sum of distances between consecutive chunks divided by the length "
            "of the shortest path. The length of the shortest path is obtained by solving the "
            "traveling salesman problem.\n\n"
        )

        # TODO: change prompt/examples
        prompt += ()

    prompt += (
        f"Sentence 1: {text}\n"
        f"Generate a sentence such that the difference in {feature} between sentence two and sentence one is {target_delta}\n"
        "Sentence 2: "
    )

    return prompt

def get_test_set(test_file, limit=100):
    with open(test_file, "r") as f:
        test_set = list(map(lambda x: x.strip().split('\t'), f.readlines()))

    return test_set[:limit]

def main():
    deltas = {
        "speed": [0.125, 0.5, 2.0, 4.0],
        "volume": [0.125, 0.5, 2.0],
        "circuitousness": [0.125, 0.5, 1.0]
    }

    for feature in ["speed", "volume", "circuitousness"]:
        for target_delta in deltas[feature]:
            test_file = f"./cev-lm/data/{feature}/data_{target_delta}_0.1/test.tsv"
            output_file = f"./gpt/results/{feature}_{target_delta}.txt"

            setup_api_key()
            test_set = get_test_set(test_file, limit=1000)

            with open(output_file, "w") as f:
                for i in tqdm(range(len(test_set))):
                    text, _ = test_set[i]
                    prompt = generate_prompt(text, target_delta, feature=feature)
                    output = run_gpt(prompt, model="davinci", temperature=0.7)
                    f.write(output.replace("\n", "") + "\n")

def test_single(text):
    feature = "speed"
    target_delta = 0.5

    setup_api_key()

    prompt = generate_prompt(text, target_delta, feature=feature)
    output = run_gpt(prompt, model="davinci", temperature=0.7)
    print(output.replace("\n", ""))
    