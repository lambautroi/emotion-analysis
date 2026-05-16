import os
import re

for file in os.listdir('.'):
    if file.endswith('.py') and file not in ('split_scripts.py', 'fix_scripts.py', 'train_model.py'):
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()

        content = re.sub(r'^[ \t]*#.*?\n', '', content, flags=re.MULTILINE)

        content = re.sub(r'[ \t]+# [^\n]*', '', content)

        content = re.sub(r'\n\s*\n', '\n\n', content)

        with open(file, 'w', encoding='utf-8') as f:
            f.write(content)
print('Comments removed from python files.')
