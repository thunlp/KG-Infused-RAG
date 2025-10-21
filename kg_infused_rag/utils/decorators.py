import time
from functools import wraps


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"⏱️ Function `{func.__name__}` took {elapsed:.2f} seconds.")
        return result
    return wrapper


def log_token_usage(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # get prompts
        if 'prompts' in kwargs:
            prompts = kwargs['prompts']
        elif len(args) >= 1:
            prompts = args[0]
        else:
            raise ValueError("Missing `prompts` parameter.")

        # get tokenizer
        if 'tokenizer' in kwargs:
            tokenizer = kwargs['tokenizer']
        elif len(args) >= 2:
            tokenizer = args[1]
        else:
            raise ValueError("Missing `tokenizer` parameter.")

        input_tokens = sum(len(tokenizer.encode(prompt)) for prompt in prompts)
        results = func(*args, **kwargs)
        output_texts = results if isinstance(results, list) else [results]
        output_tokens = sum(len(tokenizer.encode(output)) for output in output_texts)

        print(f"🧾 Input tokens: {input_tokens}")
        print(f"📝 Output tokens: {output_tokens}")
        print(f"🔢 Total tokens: {input_tokens + output_tokens}")
        return results
    return wrapper
