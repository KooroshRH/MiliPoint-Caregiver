import os
import torch
import pytorch_lightning as pl
import numpy as np
import logging


from .wrapper import ModelWrapper
from .visualize import make_video


def get_checkpoint_file(checkpoint_dir):
    for file in os.listdir(checkpoint_dir):
        if file.endswith(".ckpt"):
            return file


def plt_model_load(model, checkpoint):
    state_dict = torch.load(checkpoint)['state_dict']
    model.load_state_dict(state_dict)
    return model


def test(model, test_loader, plt_trainer_args, load_path, visualize,
         explainability=False, class_names=None, test_dataset=None, num_explain_samples=5):
    """
    Test the model and optionally run explainability analysis.

    Args:
        model: The model to test
        test_loader: DataLoader for test data
        plt_trainer_args: PyTorch Lightning trainer arguments
        load_path: Path to checkpoint to load
        visualize: Whether to create visualization video
        explainability: Whether to run explainability analysis
        class_names: List of class names for explainability visualizations
        test_dataset: Test dataset object (needed for metadata in explainability)
        num_explain_samples: Number of samples to visualize per category (TP/FN)
    """
    plt_model = ModelWrapper(model)
    if load_path is not None:
        if load_path.endswith(".ckpt"):
            checkpoint = load_path
        else:
            if load_path.endswith("/"):
                checkpoint = load_path + "best.ckpt"
            else:
                raise ValueError(
                    "if it is a directory, if must end with /; if it is a file, it must end with .ckpt")
        plt_model = plt_model_load(plt_model, checkpoint)
        plt_model.eval()
        print(f"Loaded model from {checkpoint}")

    trainer = pl.Trainer(**plt_trainer_args)
    trainer.test(plt_model, test_loader)

    if visualize:
        filename = checkpoint[:-5]+'.avi'
        logging.info(f'Saving test result in {filename}')
        res = trainer.predict(plt_model, test_loader)
        Y_pred = np.concatenate([r[0] for r in res])
        Y = np.concatenate([r[1] for r in res])
        make_video(Y_pred, Y, filename)
        logging.info(f'Saved {filename}')

    if explainability:
        logging.info("Running explainability analysis...")
        from .explainability import run_explainability_analysis

        # Determine output directory based on checkpoint path
        if load_path.endswith(".ckpt"):
            output_dir = os.path.dirname(load_path)
        else:
            output_dir = load_path.rstrip('/')
        output_dir = os.path.join(output_dir, 'explainability')

        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Run explainability analysis
        results = run_explainability_analysis(
            model=model,
            test_loader=test_loader,
            class_names=class_names,
            output_dir=output_dir,
            num_samples=num_explain_samples,
            device=device,
            test_dataset=test_dataset
        )

        logging.info(f"Explainability analysis complete:")
        logging.info(f"  - True Positives: {results['num_true_positives']}")
        logging.info(f"  - False Negatives: {results['num_false_negatives']}")
        logging.info(f"  - Accuracy: {results['accuracy']:.4f}")
        logging.info(f"  - Results saved to: {output_dir}")