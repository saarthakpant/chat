import json

with open('dataset1.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Type of data: {type(data)}")  # Should print <class 'dict'> or <class 'list'>

if isinstance(data, dict):
    print(f"Keys: {list(data.keys())}")
elif isinstance(data, list) and len(data) > 0:
    print(f"First item keys: {list(data[0].keys())}")
