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
    logging.info("="*70)
    logging.info("TEST FUNCTION CALLED")
    logging.info("="*70)
    logging.info(f"Model: {model.__class__.__name__}")
    logging.info(f"Load path: {load_path}")
    logging.info(f"Visualize: {visualize}")
    logging.info(f"Explainability: {explainability}")
    logging.info(f"Explainability samples: {num_explain_samples}")
    logging.info(f"Class names provided: {class_names is not None}")
    logging.info(f"Test dataset provided: {test_dataset is not None}")
    logging.info("="*70)

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
        logging.info("\n" + "="*70)
        logging.info("STARTING EXPLAINABILITY ANALYSIS")
        logging.info("="*70)

        try:
            from .explainability import run_explainability_analysis
            logging.info("✓ Successfully imported explainability module")
        except Exception as e:
            logging.error(f"✗ Failed to import explainability module: {e}")
            return

        # Determine output directory based on checkpoint path
        if load_path.endswith(".ckpt"):
            output_dir = os.path.dirname(load_path)
        else:
            output_dir = load_path.rstrip('/')
        output_dir = os.path.join(output_dir, 'explainability')

        logging.info(f"Output directory: {output_dir}")
        logging.info(f"Creating directory if not exists...")
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"✓ Output directory ready")

        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Device: {device}")
        logging.info(f"Number of samples to analyze: {num_explain_samples}")
        logging.info(f"Test loader batches: {len(test_loader)}")

        # Run explainability analysis
        logging.info("\nCalling run_explainability_analysis()...")
        logging.info("-"*70)

        try:
            results = run_explainability_analysis(
                model=model,
                test_loader=test_loader,
                class_names=class_names,
                output_dir=output_dir,
                num_samples=num_explain_samples,
                device=device,
                test_dataset=test_dataset
            )

            logging.info("-"*70)
            logging.info("✓ Explainability analysis completed successfully!")
            logging.info("="*70)
            logging.info("EXPLAINABILITY RESULTS:")
            logging.info("="*70)
            logging.info(f"  True Positives: {results['num_true_positives']}")
            logging.info(f"  False Negatives: {results['num_false_negatives']}")
            logging.info(f"  Accuracy: {results['accuracy']:.4f}")
            logging.info(f"  Results saved to: {output_dir}")
            logging.info("="*70)

        except Exception as e:
            logging.error("="*70)
            logging.error("✗ EXPLAINABILITY ANALYSIS FAILED")
            logging.error("="*70)
            logging.error(f"Error type: {type(e).__name__}")
            logging.error(f"Error message: {str(e)}")
            import traceback
            logging.error("Full traceback:")
            logging.error(traceback.format_exc())
            logging.error("="*70)
    else:
        logging.info("Explainability analysis not requested (explainability=False)")