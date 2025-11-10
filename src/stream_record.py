import os, time
import pandas as pd
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

from src.utils import load_cfg, makedirs

def record_task(task_name, seconds=None, out_csv=None):
    cfg = load_cfg()
    seconds = seconds or cfg["record_seconds"]

    device = cfg["device"]["type"]
    params = BrainFlowInputParams()

    if device == "muse_ble":
        params.mac_address = cfg["device"]["muse_mac"]
        board_id = BoardIds.MUSE_2_BLED_BOARD.value
        ch_names = cfg["muse_channels"]
    elif device == "openbci_serial":
        params.serial_port = cfg["device"]["serial_port"]
        board_id = BoardIds.CYTON_BOARD.value
        ch_names = cfg["openbci_channels"]
    else:
        raise ValueError("Unknown device type")

    paths = cfg["paths"]
    makedirs(paths["raw_dir"])
    out_csv = out_csv or os.path.join(paths["raw_dir"], f"eeg_{task_name}.csv")

    BoardShim.enable_dev_board_logger()
    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()

    print(f"[Recording] {task_name} for {seconds}s ...")
    time.sleep(seconds)

    data = board.get_board_data()
    board.stop_stream()
    board.release_session()

    eeg_ids = BoardShim.get_eeg_channels(board_id)
    df = pd.DataFrame(data[eeg_ids].T, columns=ch_names[:len(eeg_ids)])
    df.to_csv(out_csv, index=False)
    print(f"[Saved] {out_csv}")

if __name__ == "__main__":
    for t in ["baseline", "nback", "stroop", "breathing"]:
        record_task(t)
