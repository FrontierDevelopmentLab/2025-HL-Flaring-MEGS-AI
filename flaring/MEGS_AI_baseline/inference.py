import argparse

import pandas as pd
import torch
import numpy as np
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from SDOAIA_dataloader import AIA_GOESDataset
from models.linear_and_hybrid import HybridIrradianceModel
from callback import unnormalize_sxr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def predict_log_outputs(model, dataset, batch_size=1, times=None):
    """Generator yielding raw log-space model outputs"""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size)

    with torch.no_grad():
        for batch in loader:
            #print(batch)
            # Correct unpacking based on your data structure
            if isinstance(batch, tuple) and len(batch) == 2:
                # batch = (inputs, targets) where inputs = [aia_imgs, sxr_imgs]
                aia_imgs = batch[0][0]  # Get aia_imgs from inputs
                sxr = batch[1]
            else:
                # Fallback for other formats
                aia_imgs = batch[0][0] if isinstance(batch[0], list) else batch[0]
                sxr = batch[1].squeeze()

            # Move to device (it's already a tensor)
            aia_imgs = aia_imgs.to(device)

            # Get model predictions
            log_outputs = model(aia_imgs).squeeze()

            # Move to CPU and convert to numpy before yielding
            yield (log_outputs.cpu().numpy(), sxr.cpu().numpy())

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

    state = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    model = state['model']
    model.to(device)


    # )
    # model.to(device)

    # Assume it's a checkpoint with state_dict
    # checkpoint = torch.load(args.ckpt_path)
    # model = HybridIrradianceModel(6,1)
    # state_dict = checkpoint.get('state_dict', checkpoint)
    #
    # # Handle potential key mismatches (e.g., PyTorch Lightning prefixes)
    # state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    # model.load_state_dict(state_dict, strict=False)



    # Dataset without any output transformation
    dataset = AIA_GOESDataset(
        aia_dir=args.aia_dir,
        sxr_dir=args.sxr_dir,
    )

    df = pd.DataFrame(columns=['Timestamp','Prediction', 'Ground'])# If you have a specific normalization for SXR, load it

    times = dataset.samples
    sxr_norm = np.load(args.sxr_norm)
    # Save log-space predictions

    timestamp = []
    predictions = []
    ground = []

    for i, (log_pred, sxr) in enumerate(predict_log_outputs(model, dataset, 1, times)):
        pred = unnormalize_sxr(log_pred, sxr_norm)

        predictions.append(pred.item() if hasattr(pred, 'item') else float(pred))
        ground.append(sxr.item() if hasattr(sxr, 'item') else float(sxr))
        timestamp.append(str(times[i]))
        print(f"Processed {i+1}/{len(times)}: Timestamp={times[i]}, Prediction={pred}, Ground Truth={sxr}")



    output_df = pd.DataFrame({
        'Timestamp': timestamp,
        'Predictions': predictions,
        'ground_truth': ground
    })
    print(output_df)
    output_df.to_csv(args.output, index=False)

    #f.write(f"{times[i]},{sxr[0]},{pred[0]}\n")
    print(f"Predictions saved to {args.output}")
    # print("These are raw model outputs in log10 space before any exponentiation")

if __name__ == '__main__':
    main()