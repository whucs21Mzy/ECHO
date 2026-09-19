LLAMA2_CHAT_TEMPLATE = """{% if messages[0]['role'] == 'system' -%}
    {% set loop_messages = messages[1:] -%}
    {% set system_message = '<<SYS>>\\n' ~ messages[0]['content'] | trim ~ '\\n<</SYS>>\\n\\n' -%}
{% else -%}
    {% set loop_messages = messages -%}
    {% set system_message = '' -%}
{% endif -%}
{% for message in loop_messages -%}
    {% if loop.index0 == 0 -%}
        {% set content = system_message ~ message['content'] | trim -%}
    {% else -%}
        {% set content = message['content'] | trim -%}
    {% endif -%}
    {% if message['role'] == 'user' -%}
        {{- bos_token ~ '[INST] ' ~ content ~ ' [/INST]' -}}
    {% elif message['role'] == 'assistant' -%}
        {{- ' ' ~ content ~ ' ' ~ eos_token -}}
    {% endif -%}
{% endfor -%}"""