import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from SDOAIA_dataloader import AIA_GOESDataset
from models.linear_and_hybrid import HybridIrradianceModel
from callback import unnormalize_sxr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def predict_log_outputs(model, dataset, batch_size=8):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size)

    # Get device from model
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch in loader:
            # Correct unpacking based on your data structure
            if isinstance(batch, tuple) and len(batch) == 2:
                # batch = (inputs, targets) where inputs = [aia_imgs, sxr_imgs]
                aia_imgs = batch[0][0]  # Get aia_imgs from inputs
            else:
                # Fallback for other formats
                aia_imgs = batch[0][0] if isinstance(batch[0], list) else batch[0]

            # Move to device (it's already a tensor)
            aia_imgs = aia_imgs.to(device)

            # Get model predictions
            log_outputs = model(aia_imgs)

            # Move to CPU and convert to numpy before yielding
            yield from log_outputs.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description='Save raw log-space model outputs')
    parser.add_argument('--ckpt_path', required=True, help='Path to model checkpoint')
    parser.add_argument('--aia_dir', required=True, help='Directory of AIA images')
    parser.add_argument('--sxr_dir', required=True, help='Directory of target SXR data')
    parser.add_argument('--sxr_norm', required=True, help='Path to SXR normalization parameters (mean, std)')
    parser.add_argument('--output', default='log_predictions.txt',
                        help='Output file for log-space predictions')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Inference batch size')

    args = parser.parse_args()

    sxr_norm = np.load(args.sxr_norm)

    # Setup
    state = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    model = state['model']
    model.to(device)
    # Assume it's a checkpoint with state_dict

    # model = HybridIrradianceModel(6,1)
    # state_dict = checkpoint.get('state_dict', checkpoint)
    #
    # # Handle potential key mismatches (e.g., PyTorch Lightning prefixes)
    # state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    # model.load_state_dict(state_dict, strict=False)



    # Dataset without any output transformation
    dataset = AIA_GOESDataset(
        aia_dir=args.aia_dir,
<<<<<<< HEAD
        sxr_dir='',
        sxr_norm=None,
        transform=None
=======
        sxr_dir=args.sxr_dir,  # No SXR files needed
        transform=None  # No input transforms
>>>>>>> 22f4a17192a3a77fa4d4fe1ae3a2aa8c0bbdb539
    )

    # Save log-space predictions
    with open(args.output, 'w') as f:
        f.write("# Log-space SXR predictions (log10(W/m²))\n")
        for log_pred in predict_log_outputs(model, dataset, args.batch_size):
            pred = unnormalize_sxr(log_pred, sxr_norm)
            print(pred)

    print(f"Log-space predictions saved to {args.output}")
    print("These are raw model outputs in log10 space before any exponentiation")

if __name__ == '__main__':
    main()