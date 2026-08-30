import json

with open(r"E:\yiyi\wwwroot\python\moneyprinterturbo\topics.jsonl", encoding="utf-8") as f:
    lines = [json.loads(l) for l in f if l.strip()]

print(f"{len(lines)} tasks")
for i, t in enumerate(lines):
    print(f"{i+1}. {t['video_subject']}")
