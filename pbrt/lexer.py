import re

# Updated Token types with extended keywords for pbrt-v4.
TOKEN_TYPES = {
    'KEYWORD': r'\b(?:Camera|Shape|LightSource|AreaLightSource|Material|MakeNamedMaterial|NamedMaterial|Texture|Integrator|Sampler|PixelFilter|Film|WorldBegin|WorldEnd|AttributeBegin|AttributeEnd|Transform|ConcatTransform|LookAt|MediumInterface|MakeNamedMedium|Translate|Scale|Rotate|ActiveTransform|CoordinateSystem|CoordSysTransform)\b',
    'QUOTED': r'"[^"]*"',  # matches quoted strings
    # Use a non-greedy pattern with DOTALL to capture multiline bracketed content.
    'BRACKET': r'\[(.*?)\]',
    'NUMBER': r'[-+]?(?:\d*\.\d+|\d+\.\d*|\d+)(?:[eE][-+]?\d+)?'
}

# Combine into a single regex pattern with named groups.
TOKEN_PATTERN = re.compile(
    r'(?P<KEYWORD>' + TOKEN_TYPES['KEYWORD'] + r')'
    r'|(?P<QUOTED>' + TOKEN_TYPES['QUOTED'] + r')'
    r'|(?P<BRACKET>' + TOKEN_TYPES['BRACKET'] + r')'
    r'|(?P<NUMBER>' + TOKEN_TYPES['NUMBER'] + r')',
    re.DOTALL
)

def tokenize(text):
    tokens = []
    for match in TOKEN_PATTERN.finditer(text):
        token_type = match.lastgroup
        token_value = match.group().strip()
        tokens.append({'type': token_type, 'value': token_value})
    return tokens

def tokenize_file(filename):
    with open(filename, 'r') as f:
        text = f.read()
    return tokenize(text)
