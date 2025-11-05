import json
results = {}
for p in ['numpy','pandas','joblib','nltk','sklearn']:
    try:
        __import__(p)
        results[p] = 'OK'
    except Exception as e:
        results[p] = repr(e)
print(json.dumps(results, indent=2))
