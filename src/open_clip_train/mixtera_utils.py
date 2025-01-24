"""
Initialize Mixtera for training with WebDataset and configure parallelism.
By default, we will use one data parallel group per node. 
"""
import logging
import math
import os
import sys
import uuid
import warnings

from loguru import logger
import webdataset as wds

from mixtera.core.client import MixteraClient, ResultStreamingArgs
from mixtera.core.client.mixtera_client import QueryExecutionArgs
from mixtera.core.query import Query
from mixtera.core.query.mixture import ArbitraryMixture
from mixtera.torch import MixteraTorchDataset
from open_clip_train.distributed import world_info_from_env

warnings.simplefilter("ignore", RuntimeWarning)
warnings.simplefilter("ignore", UserWarning)

logger.remove()
# logger.add(sys.stderr, level="INFO")

def _webdataset_setup(args, is_train, epoch, floor):
    from open_clip_train.data import get_dataset_size, SharedEpoch, expand_urls

    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None

    num_shards = None
    num_shards = num_shards or len(expand_urls(input_shards)[0])

    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    'Currently, the number of dataset samples must be specified for the training dataset. '
                    'Please specify it via `--train-num-samples` if no dataset length info is present.')
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0 

    shared_epoch = SharedEpoch(epoch=epoch)

    return num_samples, num_shards, shared_epoch


def get_wds_loader(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    from open_clip_train.data import filter_no_caption_or_no_image, log_and_continue, DataInfo

    # Fetch Mixtera server info from env
    server_host = os.environ.get("MIXTERA_SERVER_ADDR", None)
    server_port = os.environ.get("MIXTERA_SERVER_PORT", None)
    job_id = os.environ.get("MIXTERA_JOB_ID", None)
    chunk_size = os.environ.get("MIXTERA_CHUNK_SIZE", 1024)

    assert server_host is not None, "MIXTERA_SERVER_ADDR must be set"
    assert server_port is not None, "MIXTERA_SERVER_PORT must be set"
    assert job_id is not None, "MIXTERA_JOB_ID must be set"

    job_id += f"_{epoch}"
    
    # Setup Mixtera
    local_rank, global_rank, world_size = world_info_from_env()
    dp_groups = world_size

    client = MixteraClient.from_remote(host=server_host, port=int(server_port))
    query = Query.for_job(job_id).select(None)
    mixture = ArbitraryMixture(chunk_size=chunk_size)

    logging.info(f"Creating Mixtera dataset with {dp_groups} data parallel groups.")

    qea = QueryExecutionArgs(
        mixture=mixture,
        num_workers=args.workers,
        dp_groups=dp_groups,
        nodes_per_group=1,
    )

    rse = ResultStreamingArgs(job_id=job_id,
                              tunnel_via_server=False,
                              dp_group_id=global_rank,
                              node_id=0
                              )


    pipeline = [
        MixteraTorchDataset(client, query, qea, rse),
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", text="txt"),
        wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0]),
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, partial=not is_train)
    ]

    dataset = wds.DataPipeline(*pipeline)

    num_samples, num_shards, shared_epoch = _webdataset_setup(args, is_train, epoch, floor)

    if is_train:
        assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    logging.info(f"Mixtera loader initialized, num_samples: {num_samples}, num_batches: {num_batches}")

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


