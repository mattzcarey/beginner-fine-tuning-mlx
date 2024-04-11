def count_iters(batch_size, train_set, epochs):
    iters_per_epoch = -(-sum(1 for _ in open(f"{train_set}/train.jsonl")) // batch_size)
    iters = iters_per_epoch * epochs

    print(f"\nAutomatically detected {iters_per_epoch} data entries.")
    print(
        f"For {epochs} epoch(s) with a batch size of {batch_size}, we will set iters to: {iters}"
    )

    return iters
