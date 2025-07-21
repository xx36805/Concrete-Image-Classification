import os
from http import HTTPStatus

import dashscope
def call_with_messages(prompt, temperature, max_n_tokens, top_p):
    messages = [
        {
            "role": "user",
            "content": [
                {"image": prompt},
                # {"text": "You are an image description assistant, please describe this specific defect image in approximately 20 words of English. Please provide a direct description and do not answer any other content."}
                {"text": "Choose the word that best describes the condition of the concrete structure in the diagram from the following five options: corrosion_of_reinforcement、crack、hole、honeycomb、normal. Please only return the closest word."},
                {"text": "Choose the word that best describes the condition of the concrete structure in the diagram from the following two options: crack、normal. Please only return the closest word."}
            ]
        }
    ]
    response = dashscope.MultiModalConversation.call(
        api_key='<YOUR API KEY>',
        model='qwen-vl-max',
        messages=messages,
        temperature=temperature,
        max_n_tokens=max_n_tokens,
        top_p=top_p,
        )
    if response.status_code == HTTPStatus.OK:
        content = response.output['choices'][0]['message']['content'][0]['text']
    else:
        content = "Sorry, I didn't understand"
    return content

if __name__ == '__main__':
    content = call_with_messages(
        "<YOU OSS PATH>",
        1, 100, 1)
    print(content)
