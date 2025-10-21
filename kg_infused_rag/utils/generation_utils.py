import os
import time
from .decorators import timeit, log_token_usage

# OpenAI
import backoff
import asyncio
from openai import OpenAI, AsyncOpenAI
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_BASE_URL"] = ""
# client = OpenAI()
client = AsyncOpenAI()


def postprocess_output(pred):
    def remove_prefix(text, prefix):
        return text[len(prefix):] if text.startswith(prefix) else text

    prefixes_to_remove = [
        "### ",
        "Note:\n",
        "Summary:\n",
        "Enhanced Passage:\n",
        "New Query:\n",
        "Answer:\n",
        "Answer: "
    ]
    pred = pred.replace("</s>", "")
    if len(pred) > 0 and pred[0] == " ":
        pred = pred[1:]
    for prefix in prefixes_to_remove:
        pred = remove_prefix(pred, prefix)
    return pred


@timeit
@log_token_usage
def call_local(prompts, tokenizer, model, sampling_params):
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    inputs = [tokenizer.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    ) for message in messages]

    preds = model.generate(inputs, sampling_params)
    preds = [pred.outputs[0].text for pred in preds]
    postprocessed_preds = [postprocess_output(pred) for pred in preds]
    return postprocessed_preds


def gpt_gen(content, model):
    try:
        completion = client.chat.completions.create(
            model=model,
            temperature=0.0,
            top_p=1.0,
            max_tokens=50,
            messages=[{"role": "user", "content": content}],
        )
        return completion.choices[0].message.content

    except Exception as e:
        print(f"An error occurred: {e}")
        time.sleep(0.5)
    return None


@backoff.on_exception(backoff.expo, (Exception), max_time=100)
def call_gpt(prompt, model):
    res = gpt_gen(prompt, model)
    assert res is not None
    return res


@backoff.on_exception(
    backoff.expo,
    Exception,  # catch all exceptions
    max_time=60,
    jitter=None
)
async def retry_single_prompt(prompt, model, tokenizer, sampling_params):
    completion = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        **sampling_params
    )
    return completion.choices[0].message.content


async def gpt_gen_batch(prompts, model, tokenizer, concurrency=5, **sampling_params):
    semaphore = asyncio.Semaphore(concurrency)

    async def safe_call(prompt):
        async with semaphore:
            try:
                return await retry_single_prompt(prompt, model, tokenizer, sampling_params)
            except Exception as e:
                print(f"[FINAL ERROR] Prompt '{prompt}' failed after retries: {e}")
                return None

    tasks = [safe_call(prompt) for prompt in prompts]
    return await asyncio.gather(*tasks)


@timeit
@log_token_usage
@backoff.on_exception(
    backoff.expo,
    Exception,  # catch all exceptions
    max_time=120,  # outer layer retries for up to 2 minutes
    jitter=None,
    on_backoff=lambda details: print(
        f"[call_gpt_batch RETRY] Attempt #{details['tries']} after exception: {details['exception']}"
    )
)
def call_gpt_batch(prompts, model, tokenizer, print_failed = True, retry_failed= True, **sampling_params):
    results = asyncio.run(
        gpt_gen_batch(prompts, model, tokenizer, **sampling_params)
    )

    # Record indices of failed items
    failed_indices = [i for i, r in enumerate(results) if r is None]

    # Optional retry
    if retry_failed and failed_indices:
        print(f"Failed Indices: {failed_indices}")
        print(f"🔁 Retrying {len(failed_indices)} failed prompts...")
        failed_prompts = [prompts[i] for i in failed_indices]
        retried_results = asyncio.run(
            gpt_gen_batch(failed_prompts, model, tokenizer, **sampling_params)
        )

        # Write back into original positions
        for idx, new_result in zip(failed_indices, retried_results):
            results[idx] = new_result

    # Optional logging of final failures
    if print_failed:
        for i, (prompt, result) in enumerate(zip(prompts, results)):
            if result is None:
                print(f"[FINAL FAIL] Prompt {i}: \n{prompt}")

    return [
        postprocess_output(r) if r is not None else ""
        for r in results
    ]
