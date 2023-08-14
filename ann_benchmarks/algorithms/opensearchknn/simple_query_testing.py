import time
from urllib3 import Timeout, PoolManager
from typing import Dict
from urllib.request import Request, urlopen

import numpy as np
import requests
import json

from opensearchpy import ConnectionError, OpenSearch
from opensearchpy.helpers import bulk
from tqdm import tqdm


def get_time():
    return round(time.time() * 1000)


def print_percentile(percentile, array, key):
    array.sort()
    print(f'{key}{percentile} : {np.percentile(array, percentile)}')


def simple_query(url):
    #session = requests.Session()
    timeout = Timeout(connect=1000.0, read=5000.0)
    http = PoolManager(timeout=timeout)

    print("Running k-NN simple queries")
    headers = {"Content-Type": "application/json"}
    total_time = []
    network_time = []
    query_time = []
    for i in range(1, 400):
        stime = get_time()
        payload = json.dumps({
            "vector": [5.2, 4.1],
            "k": 3,
            "vectorFieldName": "location",
            "initialNetworkDelay": stime
        })
        response = http.request(method='POST', url=url, body=payload, headers=headers)

        #response = requests.request("POST", url, json=payload, headers=headers)
        endTime = get_time()
        #print(response)
        res_json = json.loads(response.data.decode('utf-8'))
        #ids = [int(h["_id"]) - 1 for h in res_json["hits"]]
        #print(ids)
        # print(
        #     f'Total Time: {endTime - stime}, network: {res_json["initialNetworkDelayInMillis"]}, QueryTime: {res_json["timeInNano"] / 1000000}')
        total_time.append(endTime - stime)
        network_time.append(res_json["initialNetworkDelayInMillis"])
        query_time.append(res_json["timeInNano"] / 1000000)
    print_percentile(99, total_time, "total_time")
    print_percentile(99, network_time, "network_time")
    print_percentile(99, query_time, "query_time")

    print_percentile(90, total_time, "total_time")
    print_percentile(90, network_time, "network_time")
    print_percentile(90, query_time, "query_time")



def standard_opensearch_query():
    client = OpenSearch(["http://localhost:9200"])
    body = {
        "size": 3,
        "query": {
            "knn": {
                "location": {
                    "vector": [
                        5,
                        4
                    ],
                    "k": 3
                }
            }
        },
        "stored_fields": "_none_",
        "_source": False,
        "docvalue_fields": ["_id"]
    }
    total_time = []
    took_time = []
    for i in range(1, 400):
        stime = get_time()
        response = client.search(index='hotels-index', body=body, request_cache=False)
        etime = get_time()
        #print(f"Total time : {etime - stime} tookTime : {response['took']}")
        total_time.append(etime - stime)
        took_time.append(response['took'])
    print_percentile(99, total_time, "total_time")
    print_percentile(99, took_time, "took_time")

    print_percentile(90, total_time, "total_time")
    print_percentile(90, took_time, "took_time")


if __name__ == '__main__':
    print("-------------Standard OpenSearch-------------")
    standard_opensearch_query()
    print("-------------Simple Query-------------")
    simple_query("http://localhost:9200/vector_search/hotels-index/_query")
    print("-------------Simple Query2-------------")
    simple_query("http://localhost:9200/vector_search/hotels-index/_query2")
