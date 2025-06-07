from jinja2 import Template


def raise_exception(msg):
    """Helper used inside the Jinja template to raise runtime errors."""
    raise RuntimeError(msg)

LMSTUDIO_TEMPLATE = """
{{- bos_token }}
{%- if custom_tools is defined %}
{%- set tools = custom_tools %}
{%- endif %}
{%- if not tools_in_user_message is defined %}
{%- set tools_in_user_message = true %}
{%- endif %}
{%- if not date_string is defined %}
{%- if strftime_now is defined %}
{%- set date_string = strftime_now("%d %b %Y") %}
{%- else %}
{%- set date_string = "26 Jul 2024" %}
{%- endif %}
{%- endif %}
{%- if not tools is defined %}
{%- set tools = none %}
{%- endif %}

{# Extract system message #}
{%- if messages[0]['role'] == 'system' %}
{%- set system_message = messages[0]['content']|trim %}
{%- set messages = messages[1:] %}
{%- else %}
{%- set system_message = "" %}
{%- endif %}

{# System header #}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{%- if tools is not none %}
{{- "Environment: ipython\n" }}
{%- endif %}
{{- "Cutting Knowledge Date: December 2023\n" }}
{{- "Today Date: " + date_string + "\n\n" }}
{%- if tools is not none and not tools_in_user_message %}
{{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}
{{- 'Respond in the format {"name":function name,"parameters":{…}}. Do not use variables.\n\n' }}
{%- for t in tools %}
{{- t | tojson(indent=4) }}
{{- "\n\n" }}
{%- endfor %}
{%- endif %}
{{- system_message }}
{{- "<|eot_id|>" }}

{# Tool-calling user header #}
{%- if tools_in_user_message and not tools is none %}
{%- if messages | length != 0 %}
{%- if messages[0]['content'] is string %}
{%- set first_user_message = messages[0]['content']|trim %}
{%- else %}
{%- set first_user_message = messages[0]['content'][0]['text']|trim %}
{%- endif %}
{%- set messages = messages[1:] %}
{%- else %}
{{- raise_exception("No user message for tools!") }}
{%- endif %}
{{- '<|start_header_id|>user<|end_header_id|>\n\n' }}
{{- first_user_message + '<|eot_id|>' }}
{%- endif %}

{# Remaining messages: user, assistant, ipython, tool_results, tool_calls #}
{%- for message in messages %}
{%- if message.role == 'assistant' %}
{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' + message.content|trim + '<|eot_id|>' }}
{%- elif message.role == 'user' %}
{{- '<|start_header_id|>user<|end_header_id|>\n\n' + message.content|trim + '<|eot_id|>' }}
{%- elif 'tool_calls' in message %}
{%- set call = message.tool_calls[0].function %}
{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{{- '{"name":"' + call.name + '","parameters":' + (call.arguments|tojson) + '}' + '<|eot_id|>' }}
{%- elif message.role == 'ipython' or 'tool_results' in message.role %}
{{- '<|start_header_id|>ipython<|end_header_id|>\n\n' + (message.content|tojson if message.content is iterable else message.content) + '<|eot_id|>' }}
{%- endif %}
{%- endfor %}

{%- if add_generation_prompt %}
{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
"""

GENERATION_CONFIG = {
    "temperature": 0.8,
    "top_k": 40,
    "top_p": 0.95,
    "min_p": 0.05,
    "repeat_penalty": 1.1,
    "stop": ["<|start_header_id|>", "<|eot_id|>"]
}


def render_prompt(messages, bos_token=""):
    template = Template(
        LMSTUDIO_TEMPLATE,
        globals={"raise_exception": raise_exception},
    )
    return template.render(
        messages=messages,
        bos_token=bos_token,
        add_generation_prompt=True,
    )

