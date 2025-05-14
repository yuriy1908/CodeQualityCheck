# Отчет о качестве кода

{% for file in files %}
## Файл: {{ file.path }}

### Проблемы:
{{ file.issues }}

### Рекомендации:
{{ file.recommendations }}

{% endfor %}