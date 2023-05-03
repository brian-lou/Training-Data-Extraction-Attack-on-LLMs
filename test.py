
import openai

openai.api_key = "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"

texts = openai.Completion.create(model="text-davinci-003", 
                                             prompt="", 
                                             max_tokens=100,
                                             top_p=1.0,
                                             logprobs=1,
                                             )
print(texts)