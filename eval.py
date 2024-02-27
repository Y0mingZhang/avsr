import logging
from argparse import ArgumentParser
import time
import json
from os.path import dirname, join

import sentencepiece as spm
import torch
import torchaudio
from transforms import get_data_module
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def compute_word_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(seq1.lower().split(), seq2.lower().split())

def compute_char_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(list(seq1.lower()), list(seq2.lower()))


from lightning_av import AVConformerRNNTModule, AdditiveAVConformerRNNTModule, VotingAVConformerRNNTModule, AVBatch
from lightning import ConformerRNNTModule, Batch


MODALITY_DEFAULT_ARCHS = {
    "audio": ConformerRNNTModule,
    "video": ConformerRNNTModule,
    "audiovisual": AVConformerRNNTModule,
}

ARCHS = {
    "conformer": ConformerRNNTModule,
    "av-conformer": AVConformerRNNTModule,
    "additive-av-conformer": AdditiveAVConformerRNNTModule,
    "voting-av-conformer": VotingAVConformerRNNTModule,
}

def get_lightning_module(args):
    sp_model = spm.SentencePieceProcessor(model_file=str(args.sp_model_path))

    if not args.architecture:
        model = MODALITY_DEFAULT_ARCHS[args.modality](args, sp_model)
    else:
        model = ARCHS[args.architecture](args, sp_model)

    ckpt = torch.load(args.checkpoint_path, map_location=lambda storage, loc: storage)["state_dict"]
    model.load_state_dict(ckpt)
    model.eval()
    return model

def to_device(batch, device='cuda'):
    return type(batch)(*[t.to(device) for t in batch])


def run_eval(model, data_module):
    total_word_edit_distance = total_char_edit_distance = 0
    total_word_length = total_char_length = 0
    dataloader = data_module.test_dataloader()
    with torch.no_grad():
        for idx, (batch, sample) in enumerate(pbar := tqdm(dataloader)):
            actual = sample[0][-1]
            predicted = model(to_device(batch, model.device))
            total_word_edit_distance += compute_word_level_distance(actual, predicted)
            total_word_length += len(actual.split())

            total_char_edit_distance += compute_char_level_distance(actual, predicted)
            total_char_length += len(actual)

            pbar.set_postfix({
                "WER": total_word_edit_distance / total_word_length,
                "CER": total_char_edit_distance / total_char_length,
            })

    return  {
                "WER": total_word_edit_distance / total_word_length,
                "CER": total_char_edit_distance / total_char_length,
            }


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--modality",
        type=str,
        help="Modality",
        required=True,
    )
    parser.add_argument(
        "--architecture",
        type=str,
        help="Perform online or offline recognition.",
        default=None,
        choices=ARCHS.keys()
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="Perform online or offline recognition.",
        required=True,
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        help="Root directory to LRS3 audio-visual datasets.",
        required=True,
    )
    parser.add_argument(
        "--sp-model-path",
        type=str,
        help="Path to sentencepiece model.",
        required=True,
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        help="Path to a checkpoint model.",
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device for inference.",
        default="cuda",
    )
    parser.add_argument("--debug", action="store_true", help="whether to use debug level for logging")
    return parser.parse_args()


def init_logger(debug):
    fmt = "%(asctime)s %(message)s" if debug else "%(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")


def cli_main():
    args = parse_args()
    init_logger(args.debug)

    model = get_lightning_module(args).to(args.device)
    data_module = get_data_module(args, str(args.sp_model_path))
    t0 = time.time()
    eval_results = run_eval(model, data_module)
    inference_time = time.time() - t0

    ckpt_dir = dirname(args.checkpoint_path)
    with open(join(ckpt_dir, "eval.json"), "w") as f:
        json.dump(eval_results | {"inference-time": inference_time}, f, indent=2)

if __name__ == "__main__":
    cli_main()
