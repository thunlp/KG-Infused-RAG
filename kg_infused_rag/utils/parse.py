import re

def parse_triples(text):
    """
    Extract triples enclosed in angle brackets (<subject | predicate | object>) from the input text.

    Args:
        text (str): Input text containing triples enclosed in <...>.

    Returns:
        list: A list of triples as strings.
    """
    pattern = r"(<[^<|]+ \| [^|]+ \| [^>]+>)"
    triples = re.findall(pattern, text)    
    return triples
