import os

from pytorch_lightning import Trainer

from cifar_model import BnnOnCIFAR10


def try_all_checkpoints_in_folder(folder_path, tags_csv_path):
    trainer = Trainer()

    for f in sorted(sorted(os.listdir(folder_path)), key=len):
        if ".ckpt" in f:
            print("trying: ", f)

            model: BnnOnCIFAR10 = BnnOnCIFAR10.load_from_metrics(
                weights_path=os.path.join(folder_path, f), tags_csv=tags_csv_path
            )

            trainer.test(model)


def find_best_test_accuracy_in_logs():
    cwd = os.getcwd()

    for d in os.listdir(cwd):
        if os.path.isdir(d) and "version_" in d:
            with open(os.path.join(cwd, d, "metrics.csv")) as f:
                line = f.readlines()


def compute_accuracies_saved_models():
    # model1: BnnOnCIFAR10 = BnnOnCIFAR10.load_from_metrics()

    # model2: BnnOnCIFAR10 = BnnOnCIFAR10.load_from_metrics(
    #     weights_path="/home/nik/kth/y2/temp_data_transfer/vm1/lightning_logs/version_1/checkpoints/_ckpt_epoch_240.ckpt",
    #     tags_csv="/home/nik/kth/y2/temp_data_transfer/vm1/lightning_logs/version_1/meta_tags_for_test.csv",
    # )

    model3: BnnOnCIFAR10 = BnnOnCIFAR10.load_from_metrics(
        weights_path="/home/nik/kth/y2/temp_data_transfer/vm2/lightning_logs/version_2019-11-28_07-01-08/checkpoints/_ckpt_epoch_314.ckpt",
        tags_csv="/home/nik/kth/y2/temp_data_transfer/vm2/lightning_logs/version_2019-11-28_07-01-08/meta_tags.csv",
    )

    trainer = Trainer()

    # check the metrics.csv file in the lightning_logs
    # for the test accuracies
    trainer.test(model3)


def main():
    try_all_checkpoints_in_folder(
        "/home/nik/kth/y2/temp_data_transfer/vm2/lightning_logs/version_2019-11-28_07-01-08/checkpoints/",
        "/home/nik/kth/y2/temp_data_transfer/vm2/lightning_logs/version_2019-11-28_07-01-08/meta_tags.csv",
    )


if __name__ == "__main__":
    main()
