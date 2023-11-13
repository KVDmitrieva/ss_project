import argparse
import json
import os
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

import hw_ss.model as module_model
from hw_ss.trainer import Trainer
from hw_ss.utils import ROOT_PATH
from hw_ss.utils.object_loading import get_dataloaders
from hw_ss.utils.parse_config import ConfigParser
from hw_ss.metric import *

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, out_dir):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    # build model architecture
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    sisdr, sdr, pesq, stoi = [], [], [], []
    with torch.no_grad():
        for batch_num, batch in enumerate(tqdm(dataloaders["test"], desc="Process batch")):
            batch = Trainer.move_batch_to_device(batch, device)
            output = model(**batch)
            if type(output) is dict:
                batch.update(output)
            else:
                batch["logits"] = output

            for i in range(len(batch["signals"])):
                s = batch["signals"][i, 0]
                target = batch["target"][i]
                audio_name = batch["target_path"][i].split('/')[-1].split('.')[0]

                if len(batch["text"][i]) > 0:
                    torchaudio.save(os.path.join(out_dir, "prediction", "audio", f"{audio_name}.wav"), s.unsqueeze(0), sample_rate=16000)
                    torchaudio.save(os.path.join(out_dir, "target", "audio", f"{audio_name}.wav"), target.unsqueeze(0), sample_rate=16000)

                    with open(os.path.join(out_dir, "prediction", "transcriptions", f"{audio_name}.txt"), "xw") as f:
                        f.write(audio_name + " " + batch["text"][i])

                    with open(os.path.join(out_dir, "target", "transcriptions", f"{audio_name}.txt"), "xw") as f:
                        f.write(audio_name + " " + batch["text"][i])

                sdr.append(SDRMetric()(s, target).item())
                sisdr.append(SISDRMetric()(s, target).item())
                pesq.append(PESQMetric()(s, target).item())
                stoi.append(STOIMetric()(s, target).item())

    results = {
            "SDR": sum(sdr) / len(sdr),
            "SI-SDR": sum(sisdr) / len(sisdr),
            "PESQ": sum(pesq) / len(pesq),
            "STOI": sum(stoi) / len(stoi),
        }

    for key, val in results.items():
        print(key, val)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="output",
        type=str,
        help="Dir to write audio with texts",
    )
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=20,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    ss_model_config = Path(args.resume).parent / "config.json"
    with ss_model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    # if `--test-data-folder` was provided, set it as a default test set
    if args.test_data_folder is not None:
        test_data_folder = Path(args.test_data_folder).absolute().resolve()
        assert test_data_folder.exists()
        config.config["data"] = {
            "test": {
                "batch_size": args.batch_size,
                "num_workers": args.jobs,
                "datasets": [
                    {
                        "type": "CustomDirAudioDataset",
                        "args": {
                            "audio_dir": str(test_data_folder),
                        },
                    }
                ],
            }
        }

    assert config.config.get("data", {}).get("test", None) is not None
    config["data"]["test"]["batch_size"] = args.batch_size
    config["data"]["test"]["n_jobs"] = args.jobs

    main(config, args.output)
