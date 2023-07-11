from time import sleep
from typing import Dict
from urllib.request import Request, urlopen

import json

from opensearchpy import ConnectionError, OpenSearch
from opensearchpy.helpers import bulk
from tqdm import tqdm

client = OpenSearch(["http://localhost:9200"])

print("Running K-NN Plugin Stats API to get the Performance Stats...")
res = urlopen(Request("http://localhost:9200/_plugins/_knn/stats/?pretty"),
              timeout=20000)
response = res.read().decode("utf-8")

response_json = json.loads(response)
print(response_json["nodes"])

for nodestats in response_json["nodes"]:
    knn_stats = response_json["nodes"][nodestats]["knn_perf_stats"]
    print(response_json["nodes"][nodestats]["knn_perf_stats"])
    print(knn_stats["nmslib_latency(nanoSec)"])
    print(knn_stats["nmslib_jni_latency(nanoSec)"])
