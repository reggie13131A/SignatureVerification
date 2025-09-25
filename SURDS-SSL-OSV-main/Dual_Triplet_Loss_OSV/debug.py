from dataset import *

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser('Dual Triplet -- Evaluation | SSL for Writer Identification')
    parser.add_argument('--base_dir', type=str, default=os.getcwd())
    parser.add_argument('--dataset', type=str, default="/home/admin-ps/Documents/hatAndMask/SignatureDatasets/ChiSig")
    parser.add_argument('--saved_models', type=str, default='./saved_models')
    parser.add_argument('--load_model', type=str, default='./../Autoencoder/saved_models/BHSig260_Bengali_SSL_Encoder_RN18_AE.pth')
    parser.add_argument('--batchsize', type=int, default=1)  # change for Testing; default=16
    parser.add_argument('--print_freq', type=int,default=10)
    parser.add_argument('--learning_rate', type=float, default=1.0)
    parser.add_argument('--learning_rate_AE', type=float, default=0.005)
    parser.add_argument('--margin', type=float, default=0.2)
    parser.add_argument('--lambd', type=float, default=1.0)
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--model_path', type=str, default="/home/admin-ps/Documents/hatAndMask/signatureVerification/SURDS-SSL-OSV-main/Dual_Triplet_Loss_OSV/saved_models/DTL_ChiSig_backbone=None_RNone_C_self.pt")
    parser.add_argument('--stepsize', type=float, default=5e-5)
    parser.add_argument('--eval_type', type=str, default='self', choices=['self','cross'])
    parser.add_argument('--roc', type=bool, default=False)
    parser.add_argument('--roc_name', type=str, default=None)
    args = parser.parse_args()

    train_loader, test_loader = get_dataloader(args)
    print("==> Try iterating test_loader...")
    for i, batch in enumerate(test_loader):
        print(f"[Sample {i}] Name: {batch['img_name'][0]}, Label: {batch['label'].item()}, Writer: {batch['writer_id'][0]}")
        if i >= 100:
            break