import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")


def get_gpt_answer(msg: str):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # this is "ChatGPT" $0.002 per 1k tokens
        messages=[{"role": "user", "content": msg}],
    )

    reply_content = completion.choices[0].message.content
    return reply_content
