from pytorch_lightning import Trainer

from cifar import BnnOnCIFAR10


def main():
    model: BnnOnCIFAR10 = BnnOnCIFAR10.load_from_metrics(
        weights_path="/home/nik/kth/y2/adl/Rethinking-Binarized-Neural-Network-Optimization/data/5.2_lightning_logs/version_0/checkpoints/_ckpt_epoch_325.ckpt",
        tags_csv="/home/nik/kth/y2/adl/Rethinking-Binarized-Neural-Network-Optimization/data/5.2_lightning_logs/version_0/meta_tags.csv",
    )

    trainer = Trainer()

    trainer.test(model)


if __name__ == "__main__":
    main()
