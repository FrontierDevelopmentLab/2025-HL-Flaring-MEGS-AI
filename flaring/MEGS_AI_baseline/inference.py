import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from SDOAIA_dataloader import AIA_GOESDataset

def predict_log_outputs(model, dataset, batch_size=8):
    """Generator yielding raw log-space model outputs"""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size)

    with torch.no_grad():
        for batch in loader:
            # Handle different dataset formats
            if isinstance(batch, tuple) and len(batch) == 2:
                aia_imgs = batch[0][0]  # Unpack ((aia, sxr), target)
            else:
                aia_imgs = batch[0] if isinstance(batch, (list, tuple)) else batch

            aia_imgs = aia_imgs.to(next(model.parameters()).device)
            log_outputs = model(aia_imgs)  # Get raw log-space outputs
            yield from log_outputs.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description='Save raw log-space model outputs')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--aia-dir', required=True, help='Directory of AIA images')
    parser.add_argument('--output', default='log_predictions.txt',
                        help='Output file for log-space predictions')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Inference batch size')

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(args.model, map_location=device).to(device)

    # Dataset without any output transformation
    dataset = AIA_GOESDataset(
        aia_dir=args.aia_dir,
        sxr_dir='',  # No SXR files needed
        sxr_norm=None,  # Skip any normalization
        transform=None  # No input transforms
    )

    # Save log-space predictions
    with open(args.output, 'w') as f:
        f.write("# Log-space SXR predictions (log10(W/m²))\n")
        for log_pred in predict_log_outputs(model, dataset, args.batch_size):
            f.write(f"{log_pred:.6f}\n")  # Write with 6 decimal places

    print(f"Log-space predictions saved to {args.output}")
    print("These are raw model outputs in log10 space before any exponentiation")

if __name__ == '__main__':
    main()