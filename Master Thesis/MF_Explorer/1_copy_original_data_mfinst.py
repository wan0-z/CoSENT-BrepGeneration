from utils.get_data_mfinst import copy_labels_and_steps_only


if __name__ == "__main__":
    src_root = r"C:\Users\go36sal\Downloads\data2"
    dst_root = r"data\mfinstseg"

    n = 1000

    copy_labels_and_steps_only(
        n=n,
        src_root=src_root,
        dst_root=dst_root,
        clear_dst=True,
        require_rel_label=False,
    )