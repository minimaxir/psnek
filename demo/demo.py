"""Simulated training run showing psnek's core features, recorded by demo.tape."""

import logging
import math
import time

import psnek

EPOCHS = 3
PLATEAU_EPOCH = 2
BATCHES = 50
CHUNKS = 85
CHUNK_BYTES = 3 << 20

learning_rate = 3e-4
loss = 1.0
upload_options = psnek.Options(unit="B", scale="iec")

with psnek.Bar(total=EPOCHS, desc="train") as train:
    for epoch in range(1, EPOCHS + 1):
        batches = psnek.track(BATCHES, f"epoch {epoch}")
        # Fields set before iterating are drawn from the bar's first frame.
        batches.status(loss=f"{loss:.4f}", lr=f"{learning_rate:g}")
        for step in batches:
            time.sleep(0.02 * (1 + math.sin(step * 0.7) ** 2))
            loss = 1 / (1 + (epoch - 1) * BATCHES + step)
            if epoch == PLATEAU_EPOCH and step == BATCHES // 2:
                learning_rate = 1e-4
                train.log(
                    f"loss plateau, lowering lr to {learning_rate:g}",
                    level=logging.WARNING,
                )
            batches.status(loss=f"{loss:.4f}", lr=f"{learning_rate:g}")
        accuracy = 0.63 + 0.08 * epoch
        train.log(f"epoch {epoch}: val loss {1.4 * loss:.4f}, accuracy {accuracy:.1%}")
        train.log(f"saved checkpoint epoch-{epoch}.pt")
        train.update()
    with psnek.Bar(desc="upload", options=upload_options) as upload:
        for _ in range(CHUNKS):
            time.sleep(0.02)
            upload.update(CHUNK_BYTES)
    train.log("uploaded model.safetensors")
