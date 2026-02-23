from json import JSONDecodeError
from simpleeval import simple_eval, InvalidExpression
from ollama import chat
import json
import math

messages = []

ALGORITHM_TOOLS = {
    "gcd": (math.gcd, "inputs: [a, b]"),
    "lcm": (math.lcm, "inputs: [a, b]"),
    "factorial": (math.factorial, "inputs: [n]"),

    "is_prime": (lambda n: n > 1 and all(
        n % i != 0 for i in range(2, int(n ** 0.5) + 1)),
                 "inputs: [n]"),

    "nth_root": (lambda n, r: n ** (1 / r), "inputs: [n, root]"),
    "log": (math.log, "inputs: [n, base]"),
    "log2": (math.log2, "inputs: [n]"),
    "log10": (math.log10, "inputs: [n]"),

    "fibonacci": (lambda n: int(
        (((1 + 5 ** 0.5) / 2) ** n -
         ((1 - 5 ** 0.5) / 2) ** n) / 5 ** 0.5), "inputs: [n]"),

    "circle_area": (lambda r: math.pi * r ** 2, "inputs: [radius]"),
    "hypotenuse": (math.hypot, "inputs: [a, b]"),

    "sin": (lambda d: math.sin(math.radians(d)), "inputs: [degrees]"),
    "cos": (lambda d: math.cos(math.radians(d)), "inputs: [degrees]"),
    "tan": (lambda d: math.tan(math.radians(d)), "inputs: [degrees]"),
}

tool_descriptions = "\n".join(
    f'  - "{name}": {desc}'
    for name, (_, desc) in ALGORITHM_TOOLS.items()
)

SYSTEM_PROMPT = """You are a calculator tool-calling agent.

Rule 1: Always respond with a raw JSON object. The keys in the object depend on which tool is being called:

  - For math questions: {"tool": "calculator", "expression": "<math expression>"}
  - For unit conversions: {"tool": "converter", "value": <number>, "from_unit": "<string>", "to_unit": "<string>"}
  - For algorithm operations: {"tool": "algorithm", "operation": "<name>", "inputs": [<numbers>]}
  - For everything else: {"tool": "none", "expression": ""}
  
Rule 2: The value of "tool" must be "calculator", "converter", "algorithm", or "none".
Rule 3: The value of "expression" must be the math expression extracted from the user's message.
Rule 4: Do NOT wrap the output in markdown code fences.
Rule 5: Do NOT use ```json or ``` anywhere in your response.
Rule 6: Do NOT add any explanation, commentary, or extra text.
Rule 7: Your entire response must start with { and end with }.
Rule 8: If the user's message does not contain a math expression, you MUST respond with exactly this, no variation: {"tool": "none", "expression": ""}
Rule 9: If the user's message contains a unit conversion, respond with a JSON object 
with keys: "tool" (value: "converter"), "value" (a number), "from_unit" (a string in lowercase and always in abbreviation), 
and "to_unit" (a string in lowercase and always in abbreviation).
Rule 10: "value" must always be the numeric quantity the user wants to convert, 
extracted exactly as stated. Never substitute 0 or any default.
Rule 11: Convert natural language math to Python expressions using only operators, 
no functions. Examples:
  - "square root of 25"  → "25 ** 0.5"
  - "2 to the power of 8" → "2 ** 8"
  - "cube root of 27"    → "27 ** (1/3)"
  
Rule 12: NEVER use math functions in expressions. This includes sqrt(), pow(), abs(), 
round(), or any other function call. You MUST convert everything to operators only.
Forbidden → Allowed:
  - sqrt(25)     → 25 ** 0.5
  - pow(2, 8)    → 2 ** 8
  - abs(-5)      → (-5) if negative handled by user
  - sqrt(x**2)   → (x**2) ** 0.5
If you cannot express the operation using only +, -, *, /, **, (, ), you must not attempt it.

Rule 13: Available algorithm operations — use these exact operation names:
""" + tool_descriptions + """

Rule 14: You are ONLY allowed to use these resources:

  For "calculator" tool — these operators and nothing else:
    +  -  *  /  **  %  ( )
    No list comprehensions, no range(), no loops, no "if", no "for", 
    no Python syntax beyond simple arithmetic expressions.

  For "algorithm" tool — only the exact operation names listed in Rule 13.
    If the user's question cannot be answered using one of those operations
    or a simple arithmetic expression, respond with {"tool": "none", "expression": ""}

  If you are tempted to write anything like:
    [i for i in range(...)]  ❌
    sum([...])               ❌
    any(... for ...)         ❌
  — stop and route to "none" instead.
  
Rule 15: If a question can be answered by an algorithm operation AND a calculator 
expression, always prefer the "algorithm" tool. It is more precise.
Examples:
  - "square root of 25" → calculator (25 ** 0.5) because "nth_root" exists but 
    square root is trivially an expression
  - "GCD of 12 and 8"   → algorithm (gcd) NOT calculator, because gcd cannot be 
    expressed as a single arithmetic expression without loops
  - "factorial of 5"    → algorithm (factorial) NOT "5*4*3*2*1" as a calculator 
    expression, because the algorithm tool handles it exactly

Rule 16: Never infer, assume, or hallucinate inputs that the user did not explicitly 
state. If a required input is missing or ambiguous, respond with {"tool": "none", 
"expression": ""} instead of guessing.
Examples of what NOT to do:
  - User says "find the log" with no number     → do NOT assume log(10) ❌
  - User says "convert km to miles" with no value → do NOT assume value: 1 ❌  
  - User says "is it prime" with no number      → do NOT assume any number ❌
  - User says "common divisibles of 21" with only one number → 
    gcd requires TWO numbers → respond with {"tool": "none", "expression": ""}  ❌ do NOT call gcd([21])
The only valid response when inputs are missing is {"tool": "none", "expression": ""}

Algorithm input: "What is the GCD of 144 and 49?"
Required output: {"tool": "algorithm", "operation": "gcd", "inputs": [144, 49]}

Algorithm input: "Is 17 prime?"
Required output: {"tool": "algorithm", "operation": "is_prime", "inputs": [17]} 

Non-math input: "What is the capital of France?"
Required output: {"tool": "none", "expression": ""}

Example input: "What is the square root of 25?"
Example output: {"tool": "calculator", "expression": "25 ** 0.5"}

Conversion input: ""Convert 100 km to miles""
Required output: {"tool": "converter", "value": 100, "from_unit": "km", "to_unit": "miles"}

Example input: "What is 10 + 5?"
Example output: {"tool": "calculator", "expression": "10 + 5"}

Conversion input: "I have 32 degrees fahrenheit, what is that in celsius?"
Required output: {"tool": "converter", "value": 32, "from_unit": "fahrenheit", "to_unit": "celsius"}

"""



def parse_response(s):
    try:
        return json.loads(s)
    except JSONDecodeError:
        print('Error: Could not decode model response as JSON')
        return None


def calculate(parsed):
    tool = parsed['tool']
    expression = parsed['expression']
    print(f'Calling tool: {tool}')
    print(f'Expression: {expression}')
    try:
        result = simple_eval(expression)
        return f'Result: {result:.2f}'

    except ZeroDivisionError:
        print('Error: Division by zero is not allowed')
        return None

    except InvalidExpression:
        print('Error: Could not evaluate expression')
        return None

    except Exception as e:
        print(f'Error: Unexpected error — {e}')
        return None


def convert(parsed):
    tool = parsed['tool']
    value = parsed['value']
    from_unit = parsed['from_unit']
    to_unit = parsed['to_unit']
    print(f'Calling tool: {tool}')
    print(f'Converting : {value} {from_unit} to {to_unit}')

    new_value = 0
    match from_unit + '_' + to_unit:
        case 'km_miles':
            new_value = value * 0.621371
        case 'miles_km':
            new_value = value * 1.60934
        case 'kg_lbs':
            new_value = value * 2.20462
        case 'lbs_kg':
            new_value = value * 0.453592
        case 'celsius_fahrenheit':
            new_value = (value * 9 / 5) + 32
        case 'fahrenheit_celsius':
            new_value = (value - 32) * 5 / 9
        case _:
            print(f'Error: Unsupported conversion — {from_unit} to {to_unit}')
            return None

    return f'Result : {value} {from_unit} = {new_value:.2f} {to_unit}'


def run_algorithm(parsed):
    operation = parsed['operation']
    inputs = parsed['inputs']

    if operation not in ALGORITHM_TOOLS:
        print(f'Error: Unknown operation — {operation}')
        return None

    func, desc = ALGORITHM_TOOLS[operation]          # ← get expected input count
    expected = int(desc.split('[')[1].rstrip(']').count(',')) + 1

    if len(inputs) != expected:
        print(f'Error: {operation} expects {expected} input(s), got {len(inputs)}')
        return None

    print(f'Performing {operation} on {inputs}')
    try:
        result = func(*inputs)
        return f'Result: {operation}({", ".join(map(str, inputs))}) = {result}'
    except Exception as e:
        print(f'Error: {e}')
        return None

messages.append({'role': 'system', 'content': SYSTEM_PROMPT})

while True:
    user_input = input("\nAsk a math question (or type 'exit'): ")
    if user_input.lower() == 'exit':
        break

    message = messages + [{'role': 'user', 'content': user_input}]
    stream = chat(model='llama3.1', messages=message)

    parsed = parse_response(stream['message']['content'])

    result = ''

    if parsed is not None:
        if parsed['tool'] == 'none':
            print("I can only handle math questions and conversions")
        elif parsed['tool'].lower() == 'calculator':
            result = calculate(parsed)
        elif parsed['tool'].lower() == 'converter':
            result = convert(parsed)
        elif parsed['tool'].lower() == 'algorithm':
            result = run_algorithm(parsed)

    if result:
        RESPONSE_PROMPT = "You are a helpful assistant. Answer naturally in one sentence. Do not use JSON."

        message = [
            {'role': 'system', 'content': RESPONSE_PROMPT},
            {'role': 'user', 'content': user_input},
            {'role': 'assistant', 'content': f'Tool result: {result}'},
            {'role': 'user', 'content': 'Now summarize that result naturally in one sentence.'}
        ]
        stream = chat(model='llama3.1', messages=message, stream=True)

        for chunks in stream:
            print(chunks['message']['content'], end='', flush=True)
        print()
