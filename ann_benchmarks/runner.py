import argparse
import json
import logging
import os
import threading
import time
import traceback

import colors
import docker
import numpy
import psutil
import re

from .definitions import Definition, instantiate_algorithm
from .datasets import DATASETS, get_dataset
from .distance import dataset_transform, metrics
from .results import store_results
import numpy as np


def run_individual_query(algo, X_train, X_test, distance, count, run_count, batch):
    prepared_queries = (batch and hasattr(algo, "prepare_batch_query")) or (
            (not batch) and hasattr(algo, "prepare_query")
    )
    query_stats = []
    best_search_time = float("inf")
    for i in range(run_count):
        print("Run %d/%d..." % (i + 1, run_count))
        # a bit dumb but can't be a scalar since of Python's scoping rules
        n_items_processed = [0]

        def single_query(v):
            if prepared_queries:
                algo.prepare_query(v, count)
                start = time.time()
                algo.run_prepared_query()
                total = time.time() - start
                candidates = algo.get_prepared_query_results()
            else:
                start = time.time()
                candidates = algo.query(v, count)
                total = time.time() - start
            candidates = [
                (int(idx), float(metrics[distance]["distance"](v, X_train[idx]))) for idx in candidates  # noqa
            ]
            n_items_processed[0] += 1
            if n_items_processed[0] % 1000 == 0:
                print("Processed %d/%d queries..." % (n_items_processed[0], len(X_test)))
            if len(candidates) > count:
                print(
                    "warning: algorithm %s returned %d results, but count"
                    " is only %d)" % (algo, len(candidates), count)
                )
            return (total, candidates)

        def batch_query(X):
            if prepared_queries:
                algo.prepare_batch_query(X, count)
                start = time.time()
                algo.run_batch_query()
                total = time.time() - start
            else:
                start = time.time()
                algo.batch_query(X, count)
                total = time.time() - start
            results = algo.get_batch_results()
            candidates = [
                [(int(idx), float(metrics[distance]["distance"](v, X_train[idx]))) for idx in single_results]  # noqa
                for v, single_results in zip(X, results)
            ]
            return [(total / float(len(X)), v) for v in candidates]

        if batch:
            results = batch_query(X_test)
        else:
            results = [single_query(x) for x in X_test]

        query_times = []
        for time2, _ in results:
            query_times.append(time2 * 1000)

        query_stat = {
            "min": min(time3 * 1000 for time3, _ in results),
            "max": max(time4 * 100 for time4, _ in results),
            "avg": np.average(query_times),
            "p50": np.percentile(query_times, 50),
            "p90": np.percentile(query_times, 90),
            "p99": np.percentile(query_times, 99)
        }
        query_stats.append(query_stat)
        print(f"Query stats are {query_stat}")

        total_time = sum(time for time, _ in results)
        total_candidates = sum(len(candidates) for _, candidates in results)
        search_time = total_time / len(X_test)
        avg_candidates = total_candidates / len(X_test)
        best_search_time = min(best_search_time, search_time)

    verbose = hasattr(algo, "query_verbose")
    attrs = {
        "batch_mode": batch,
        "best_search_time": best_search_time,
        "candidates": avg_candidates,
        "expect_extra": verbose,
        "name": str(algo),
        "run_count": run_count,
        "distance": distance,
        "count": int(count),
    }
    additional = algo.get_additional()
    for k in additional:
        attrs[k] = additional[k]
    return attrs, results, query_stats


def get_stats_descriptor(algo, algo_name):
    stats = algo.get_stats()
    if algo_name != "opensearchknn":
        return [], []

    nodes_stats = stats["nodes"]
    nmslib = []
    nmslib_jni = []
    for node in nodes_stats:
        knn_stats = nodes_stats[node]["knn_perf_stats"]
        nmslib.append(knn_stats["nmslib_latency(nanoSec)"])
        nmslib_jni.append(knn_stats["nmslib_jni_latency(nanoSec)"])
    return nmslib, nmslib_jni


def run(definition, dataset, count, run_count, batch):
    algo = instantiate_algorithm(definition)
    assert not definition.query_argument_groups or hasattr(
        algo, "set_query_arguments"
    ), """\
error: query argument groups have been specified for %s.%s(%s), but the \
algorithm instantiated from it does not implement the set_query_arguments \
function""" % (
        definition.module,
        definition.constructor,
        definition.arguments,
    )

    D, dimension = get_dataset(dataset)
    X_train = numpy.array(D["train"])
    X_test = numpy.array(D["test"])
    distance = D.attrs["distance"]
    print("got a train set of size (%d * %d)" % (X_train.shape[0], dimension))
    print("got %d queries" % len(X_test))

    X_train, X_test = dataset_transform(D)

    try:
        if hasattr(algo, "supports_prepared_queries"):
            algo.supports_prepared_queries()

        t0 = time.time()
        memory_usage_before = algo.get_memory_usage()
        algo.fit(X_train)
        build_time = time.time() - t0
        index_size = algo.get_memory_usage() - memory_usage_before
        print("Built index in", build_time)
        print("Index size: ", index_size)

        query_argument_groups = definition.query_argument_groups
        # Make sure that algorithms with no query argument groups still get run
        # once by providing them with a single, empty, harmless group
        if not query_argument_groups:
            query_argument_groups = [[]]

        for pos, query_arguments in enumerate(query_argument_groups, 1):
            print("Running query argument group %d of %d..." % (pos, len(query_argument_groups)))
            if query_arguments:
                algo.set_query_arguments(*query_arguments)
            descriptor, results, query_stats = run_individual_query(algo, X_train, X_test, distance, count, run_count,
                                                                    batch)
            nmslib, nmslib_jni = get_stats_descriptor(algo, definition.algorithm)
            descriptor["build_time"] = build_time
            descriptor["index_size"] = index_size
            descriptor["algo"] = definition.algorithm
            descriptor["dataset"] = dataset
            store_results(dataset, count, definition, query_arguments, descriptor, results, batch)
            if definition.algorithm is "opensearchknn":
                write_stats_full_in_file(dataset, query_arguments, definition, query_stats, nmslib, nmslib_jni)
                write_stats_important_stats_in_file(dataset, query_arguments, definition, query_stats, nmslib, nmslib_jni)
    finally:
        algo.done()


def write_stats_full_in_file(dataset, query_arguments, definition, query_stats, nmslib, nmslib_jni):
    d = ["results"]
    if dataset:
        d.append(dataset)
    if definition:
        data = definition.arguments + query_arguments
        d.append(re.sub(r"\W+", "_", json.dumps(data, sort_keys=True)).strip("_") + ".csv")
    file_name = os.path.join(*d)
    file = open(file_name, 'w')
    lines = []
    lines.append("query(min)\tquery(max)\tquery(avg)\tquery(p50)\tquery(p90)\tquery(p99)")
    lines.append("\n")
    print("query(min)\tquery(max)\tquery(avg)\tquery(p50)\tquery(p90)\tquery(p99)")
    lines.append("\n")
    for q_stat in query_stats:
        print(f'{q_stat["min"]}\t{q_stat["max"]}\t{q_stat["avg"]}\t{q_stat["p50"]}\t{q_stat["p90"]}\t{q_stat["p99"]}')
        lines.append(f'{q_stat["min"]}\t{q_stat["max"]}\t{q_stat["avg"]}\t{q_stat["p50"]}\t{q_stat["p90"]}\t{q_stat["p99"]}')
        lines.append("\n")

    print(
        "nmslib_jni(min)\tnmslib_jni(max)\tnmslib_jni(avg)\tnmslib_jni(p50)\tnmslib_jni(p90)\tnmslib_jni(p99)")
    lines.append("nmslib_jni(min)\tnmslib_jni(max)\tnmslib_jni(avg)\tnmslib_jni(p50)\tnmslib_jni(p90)\tnmslib_jni(p99)")
    lines.append("\n")

    for lib_stat in nmslib_jni:
        print(
            f'{lib_stat["min"]}\t{lib_stat["max"]}\t{lib_stat["average"]}\t{lib_stat["p50"]}\t{lib_stat["p90"]}\t{lib_stat["p99"]}')
        lines.append(
            f'{lib_stat["min"]}\t{lib_stat["max"]}\t{lib_stat["average"]}\t{lib_stat["p50"]}\t{lib_stat["p90"]}\t{lib_stat["p99"]}')
        lines.append("\n")

    print("nmslib(min)\tnmslib(max)\tnmslib(avg)\tnmslib(p50)\tnmslib(p90)\tnmslib(p99)")
    lines.append("nmslib(min)\tnmslib(max)\tnmslib(avg)\tnmslib(p50)\tnmslib(p90)\tnmslib(p99)")
    lines.append("\n")

    for lib_stat in nmslib:
        print(
            f'{lib_stat["min"]}\t{lib_stat["max"]}\t{lib_stat["average"]}\t{lib_stat["p50"]}\t{lib_stat["p90"]}\t{lib_stat["p99"]}')
        lines.append(f'{lib_stat["min"]}\t{lib_stat["max"]}\t{lib_stat["average"]}\t{lib_stat["p50"]}\t{lib_stat["p90"]}\t{lib_stat["p99"]}')
        lines.append("\n")

    file.writelines(lines)
    file.close()


def write_stats_important_stats_in_file(dataset, query_arguments, definition, query_stats, nmslib, nmslib_jni):
    d = ["results"]
    if dataset:
        d.append(dataset)
    if definition:
        data = definition.arguments + query_arguments
        d.append(re.sub(r"\W+", "_", json.dumps(data, sort_keys=True)).strip("_") + "_important_stats.csv")
    file_name = os.path.join(*d)
    file = open(file_name, 'w')
    lines = []
    lines.append("query(p90),query(p99),nmslib_jni(p90),nmslib_jni(p99),nmslib(p90),nmslib(p99)")
    lines.append("\n")
    metrics_string = ""
    for q_stat in query_stats:
        metrics_string = metrics_string + f'{q_stat["p90"]},{q_stat["p99"]},'

    for jni_stat in nmslib_jni:
        metrics_string = metrics_string + f'{float(jni_stat["p90"])/1000000},{float(jni_stat["p99"])/1000000},'

    for lib_stat in nmslib:
        metrics_string = metrics_string + f'{float(lib_stat["p90"])/1000000},{float(lib_stat["p99"])/1000000},'

    lines.append(metrics_string)
    print("query(p90),query(p99),nmslib_jni(p90),nmslib_jni(p99),nmslib(p90),nmslib(p99)")
    print(metrics_string)
    lines.append("\n")
    file.writelines(lines)
    file.close()


def run_from_cmdline():
    parser = argparse.ArgumentParser(
        """

            NOTICE: You probably want to run.py rather than this script.

"""
    )
    parser.add_argument("--dataset", choices=DATASETS.keys(), help="Dataset to benchmark on.", required=True)
    parser.add_argument("--algorithm", help="Name of algorithm for saving the results.", required=True)
    parser.add_argument(
        "--module", help='Python module containing algorithm. E.g. "ann_benchmarks.algorithms.annoy"', required=True
    )
    parser.add_argument("--constructor", help='Constructer to load from modulel. E.g. "Annoy"', required=True)
    parser.add_argument(
        "--count", help="K: Number of nearest neighbours for the algorithm to return.", required=True, type=int
    )
    parser.add_argument(
        "--runs",
        help="Number of times to run the algorihm. Will use the fastest run-time over the bunch.",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--batch",
        help='If flag included, algorithms will be run in batch mode, rather than "individual query" mode.',
        action="store_true",
    )
    parser.add_argument("build", help='JSON of arguments to pass to the constructor. E.g. ["angular", 100]')
    parser.add_argument("queries", help="JSON of arguments to pass to the queries. E.g. [100]", nargs="*", default=[])
    args = parser.parse_args()
    algo_args = json.loads(args.build)
    print(algo_args)
    query_args = [json.loads(q) for q in args.queries]

    definition = Definition(
        algorithm=args.algorithm,
        docker_tag=None,  # not needed
        module=args.module,
        constructor=args.constructor,
        arguments=algo_args,
        query_argument_groups=query_args,
        disabled=False,
    )
    run(definition, args.dataset, args.count, args.runs, args.batch)


def run_docker(definition, dataset, count, runs, timeout, batch, cpu_limit, mem_limit=None):
    cmd = [
        "--dataset",
        dataset,
        "--algorithm",
        definition.algorithm,
        "--module",
        definition.module,
        "--constructor",
        definition.constructor,
        "--runs",
        str(runs),
        "--count",
        str(count),
    ]
    if batch:
        cmd += ["--batch"]
    cmd.append(json.dumps(definition.arguments))
    cmd += [json.dumps(qag) for qag in definition.query_argument_groups]

    client = docker.from_env()
    mem_limit = 17179869184  # 16GB

    container = client.containers.run(
        definition.docker_tag,
        cmd,
        volumes={
            os.path.abspath("ann_benchmarks"): {"bind": "/home/app/ann_benchmarks", "mode": "ro"},
            os.path.abspath("data"): {"bind": "/home/app/data", "mode": "ro"},
            os.path.abspath("results"): {"bind": "/home/app/results", "mode": "rw"},
            os.path.abspath("logs"): {"bind": "/home/app/opensearch/logs", "mode": "rw"},
        },
        cpuset_cpus=cpu_limit,
        mem_limit=mem_limit,
        detach=True,
    )
    logger = logging.getLogger(f"annb.{container.short_id}")

    logger.info(
        "Created container %s: CPU limit %s, mem limit %s, timeout %d, command %s"
        % (container.short_id, cpu_limit, mem_limit, timeout, cmd)
    )

    def stream_logs():
        for line in container.logs(stream=True):
            logger.info(colors.color(line.decode().rstrip(), fg="blue"))

    t = threading.Thread(target=stream_logs, daemon=True)
    t.start()

    try:
        return_value = container.wait(timeout=timeout)
        _handle_container_return_value(return_value, container, logger)
    except:
        logger.error("Container.wait for container %s failed with exception" % container.short_id)
        traceback.print_exc()
    finally:
        logger.info("Removing container")
        # container.remove(force=True)


def _handle_container_return_value(return_value, container, logger):
    base_msg = "Child process for container %s" % (container.short_id)
    if type(return_value) is dict:  # The return value from container.wait changes from int to dict in docker 3.0.0
        error_msg = return_value.get("Error")
        exit_code = return_value["StatusCode"]
        msg = base_msg + "returned exit code %d with message %s" % (exit_code, error_msg)
    else:
        exit_code = return_value
        msg = base_msg + "returned exit code %d" % (exit_code)

    if exit_code not in [0, None]:
        logger.error(colors.color(container.logs().decode(), fg="red"))
        logger.error(msg)
