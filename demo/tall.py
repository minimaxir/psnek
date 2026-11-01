"""Simulated nightly search-index refresh in two-row bars, recorded by tall.tape."""

import logging
import math
import time
from dataclasses import replace

import psnek

PAGES = 52
PAGE_SIZE = 240
TICKETS = PAGES * PAGE_SIZE
BATCH = 96
TOKENS_PER_TICKET = 412
DOLLARS_PER_MILLION_TOKENS = 0.13
OVERLONG_BATCH = 61
GRAPH_LAYERS = 5
TARGET = "tickets-v43"
QUERY_SETS = (
    ("billing", 420, 0.973, 2.8),
    ("login", 380, 0.961, 3.1),
    ("shipping", 510, 0.948, 3.4),
    ("refunds", 290, 0.982, 2.6),
)

stage = psnek.Options(height=2, leave=True, log_time=True)
rollout = replace(stage, leave=None, gradient=("#E5534B", "#E3B341", "#1B976F"))

with psnek.Bar(total=4, desc="refresh support search", options=rollout) as refresh:
    refresh.status(target=TARGET)

    with psnek.Bar(desc="fetch new tickets", options=stage) as fetch:
        fetch.status(api="helpdesk", since="2026-10-26", page=0)
        for page in range(1, PAGES + 1):
            time.sleep(0.03 * (1 + math.cos(page * 0.5) ** 2))
            fetch.update(PAGE_SIZE)
            fetch.status(api="helpdesk", since="2026-10-26", page=page)
    refresh.update()

    tokens = 0
    with psnek.Bar(total=TICKETS, desc="embed tickets", options=stage) as embed:
        embed.status(model="text-embedding-3-small", tokens="0.00M", cost="$0.00")
        for batch, start in enumerate(range(0, TICKETS, BATCH), start=1):
            time.sleep(0.012 * (1 + math.sin(batch * 0.9) ** 2))
            size = min(BATCH, TICKETS - start)
            tokens += TOKENS_PER_TICKET * size
            if batch == OVERLONG_BATCH:
                embed.log(
                    f"batch {batch}: 2 tickets over 8,191 tokens, truncated",
                    level=logging.WARNING,
                )
            embed.update(size)
            embed.status(
                model="text-embedding-3-small",
                tokens=f"{tokens / 1e6:.2f}M",
                cost=f"${tokens / 1e6 * DOLLARS_PER_MILLION_TOKENS:.2f}",
            )
    refresh.update()

    with psnek.Bar(total=TICKETS, desc="build HNSW index", options=stage) as index:
        index.status(layer=f"0/{GRAPH_LAYERS}", M=16, ef_construction=200)
        for layer in range(1, GRAPH_LAYERS + 1):
            for _ in range(TICKETS // GRAPH_LAYERS // BATCH):
                time.sleep(0.012)
                index.update(BATCH)
            index.status(layer=f"{layer}/{GRAPH_LAYERS}", M=16, ef_construction=200)
    refresh.update()

    results = []
    evaluate = psnek.track(QUERY_SETS, "evaluate recall", options=stage)
    evaluate.status(k=10, query_set=QUERY_SETS[0][0])
    for name, queries, recall, latency in evaluate:
        evaluate.status(k=10, query_set=name)
        time.sleep(0.5)
        results.append(
            {
                "query set": name,
                "queries": queries,
                "recall@10": f"{recall:.1%}",
                "p50": f"{latency:.1f}ms",
            }
        )
        refresh.panel(results)
    refresh.update()
    refresh.log(f"{TARGET} passed recall checks, swapped into production")
