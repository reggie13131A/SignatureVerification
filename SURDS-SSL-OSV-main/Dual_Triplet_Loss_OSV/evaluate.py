import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import torch
from torch.nn.functional import mse_loss
from torchvision.utils import save_image
import random
import json
# from sklearn_extra.cluster import KMedoids
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC as SVM
from dataset import *
from model import *
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(1)
random.seed(1)
np.random.seed(1)

# def plot_roc(tpr, fpr):
#     assert len(tpr) == len(fpr)
#     plt.plot(fpr, tpr, marker='.')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.savefig("./ROC.png", dpi=300) 
    
### Taken from SigNet paper
def compute_accuracy_roc(predictions, labels, step=None):
    dmax = np.max(predictions)
    dmin = np.min(predictions)
    nsame = np.sum(labels == 1)
    ndiff = np.sum(labels == 0)
    if step is None:
        step = 0.00005

    max_acc, min_frr, min_far = 0.0, 1.0, 1.0
    d_optimal = 0.0
    tpr_arr, far_arr = [], []
    for d in np.arange(dmin, dmax +   step, step):
        idx1 = predictions.ravel () <= d     # pred = 1
        idx2 = predictions.ravel() > d      # pred = 0

        tpr = float(np.sum(labels[idx1] == 1)) / nsame
        tnr = float(np.sum(labels[idx2] == 0)) / ndiff

        frr = float(np.sum(labels[idx2] == 1)) / nsame
        far = float(np.sum(labels[idx1] == 0)) / ndiff

        tpr_arr.append(tpr)
        far_arr.append(far)

        acc = 0.5 * (tpr + tnr)
        
        # print(f"Threshold = {d} | Accuracy = {acc:.4f}")

        if acc > max_acc:
            max_acc = acc
            d_optimal = d
            
            # FRR, FAR metrics
            min_frr = float(np.sum(labels[idx2] == 1)) / nsame
            min_far = float(np.sum(labels[idx1] == 0)) / ndiff
            
    
    metrics = {"best_acc" : max_acc, "best_frr" : min_frr, "best_far" : min_far, "tpr_arr" : tpr_arr, "far_arr" : far_arr}
    return metrics, d_optimal


def plot_roc(tpr, fpr, fname):
    assert len(tpr) == len(fpr)
    plt.plot(fpr, tpr, marker='.')
    plt.plot(fpr, fpr, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(f"./ROC_{fname}.png", dpi=300)
    
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
    # parser.add_argument('--model_path', type=str, default='./saved_models/BHSig260_Bengali_200epochs_LR=0.1_LR-AE=0.0001_LBD=1.0_backbone=AE.pt')
    parser.add_argument('--model_path', type=str, default="/home/admin-ps/Documents/hatAndMask/signatureVerification/SURDS-SSL-OSV-main/Dual_Triplet_Loss_OSV/saved_models/DTL_ChiSig_backbone=None_RNone_C_self.pt")
    parser.add_argument('--stepsize', type=float, default=5e-5)
    parser.add_argument('--eval_type', type=str, default='self', choices=['self','cross'])
    parser.add_argument('--roc', type=bool, default=False)
    parser.add_argument('--roc_name', type=str, default=None)
    args = parser.parse_args()

    print('\n'+'*'*100)

    # 1. get data
    train_loader, test_loader = get_dataloader(args)
    
    # 2. load model
    MODEL_PATH = args.model_path
    # THRESHOLD = 0.001934

    checkpoint = torch.load(MODEL_PATH)
    print(f"Loading model from: {MODEL_PATH} | Epochs trained: {checkpoint['epochs']}")
    model = Triplet_Model(args)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    # 3. feature extraction from train and test
    features, labels, writer_ids, img_names = [], [], [], []
    mean_AP, mean_AN = 0.0, 0.0

    with torch.no_grad():
        if args.eval_type == 'cross':
            for batch in train_loader:
                feat = model.projector(model.encoder(batch['anchor'].to(device), pool=True))
                # P_feat = model.projector(model.encoder(batch['positive'].to(device), pool=True))
                # N_feat = model.projector(model.encoder(batch['negative_intra'].to(device), pool=True))
                # mean_AP += np.abs(np.mean(np.subtract(feat.cpu().numpy(), P_feat.cpu().numpy())))
                # mean_AN += np.abs(np.mean(np.subtract(feat.cpu().numpy(), N_feat.cpu().numpy())))
                features.append(feat.cpu().numpy())
                labels.append(batch['label'].cpu().numpy())
                writer_ids.append(int(batch['writer_id'][0]))
                img_names.append(batch['img_name'][0])
            # mean_AP /= len(train_loader)
            # mean_AN /= len(train_loader)
            # print(len(features), len(labels), len(writer_ids))
            # print(f"Train set: Mean A--P distance = {mean_AP} | Mean A--N distance = {mean_AN}")
        for batch in test_loader:
            feat = model.projector(model.encoder(batch['image'].to(device), pool=True))
            features.append(feat.cpu().numpy())
            labels.append(batch['label'].cpu().numpy())
            # writer_ids.append(int(batch['writer_id'][0]))
            writer_ids.append(batch['writer_id'][0])
            img_names.append(batch['img_name'][0])
        # print(len(features), len(labels), len(writer_ids), len(img_names))
        print(writer_ids[:10])

    features = np.array(features)
    labels = np.array(labels)
    # ???
    writer_ids = np.array(writer_ids)
    print(writer_ids[:10])

    print(features.shape, labels.shape)
    n_samples = features.shape[0] * features.shape[1]
    n_features = features.shape[2]

    # 挤掉多余的dim
    features = features.reshape(n_samples, n_features)
    labels = labels.reshape(labels.shape[0])
    print(features.shape, labels.shape)

    X = features.copy()
    y = labels.copy()
    W_id = writer_ids.copy()
    X_train, X_test, y_train, y_test, Wid_train, Wid_test = train_test_split(X, y, W_id, test_size=0.3, shuffle=False, random_state=1)

    df = pd.DataFrame(features)
    df['label'] = labels.copy()
    df['writer_id'] = writer_ids.copy()
    df['img_name'] = img_names.copy()
    print(df.head())

    # 创建一个新的集合
    wrtr_set = set()
    df_ref_writer = pd.DataFrame()

    for i in range(len(df)):
        label = df.iloc[i]['label']
        writer = df.iloc[i]['writer_id']
        img_name = df.iloc[i]['img_name']

        if writer not in wrtr_set:
            wrtr_set.add(writer)
            print(f">> Creating reference set for Writer ID: {writer}")
            # 提取出标签为1的样本作为参考
            df_ref = df[(df['writer_id']==writer) & (df['label']==1)]
            assert (len(df_ref) == 5), f"错误：作家 {writer} 只有 {len(df_ref)} 个真实签名"
            df_ref = df_ref[(df_ref['img_name'] != img_name)]
            # 选择4个真实签名
            df_ref = df_ref.sample(4, random_state=0)  
            assert (len(df_ref) == 4)
            df_ref_writer = df_ref_writer._append(df_ref)

    print(f"reference set中一共有: {len(df_ref_writer)} 张图片")
    wrong_label_refs = df_ref_writer[df_ref_writer['label'] != 1]
    if len(wrong_label_refs) > 0:
        print(f"❌ 有 {len(wrong_label_refs)} 张参考图像不是 label=1 的真实签名：")
        print(wrong_label_refs)
    else:
        print("✅ 所有参考图像均为真实签名 (label=1)")

    grouped = df_ref_writer.groupby('writer_id')
    with open('reference_images.txt', 'w', encoding='utf-8') as f:
        for writer, group in grouped:
            f.write(f"\n📌 Writer ID: {writer} | 参考图像数量: {len(group)}\n")
            f.write("参考图像 img_name:\n")
            for img in group['img_name'].values:
                f.write(f"  - {img}\n")

    dist, y_true, identify_writer = [], [], []

    preds = pd.DataFrame(columns=['img_name', 'writer_id', 'y_true', 'y_pred'])
    df_ref_writer['img_name'] = df_ref_writer['img_name'].astype(str).str.strip()
    df['img_name'] = df['img_name'].astype(str).str.strip()
    ref_img_set = set(df_ref_writer['img_name'].apply(lambda x: str(x).strip()).tolist())

    for i in range(len(df)):
        feature = np.array(df.iloc[i][0:1536]).flatten()   # 这里是将object的
        label = df.iloc[i]['label']
        writer = df.iloc[i]['writer_id']
        img_name = str(df.iloc[i]['img_name']).strip()
        # img_name = df.iloc[i]['img_name']
        
        # if 当前index的文件名不在参考集里面
        # if img_name not in set(list(df_ref_writer['img_name'])):
        if img_name not in ref_img_set:
            identify_writer.append(img_name)
            df_ref = df_ref_writer[(df_ref_writer['writer_id']==writer)]
            assert (len(df_ref) == 4)
            df_ref = df_ref.drop(['label', 'writer_id', 'img_name'], axis=1)
            mean_ref = np.mean(np.array(df_ref, dtype=np.float32), axis=0)
            # ??? 
            # mse_diff = np.abs(np.mean(np.subtract(feature, mean_ref)))
            mse_diff = np.mean((feature - mean_ref) ** 2)
            # y_pred = 1 if mse_diff <= THRESHOLD else 0
            # preds = preds.append({'img_name' : img_name, 'writer_id' : writer, 'y_true' : label, 'y_pred' : y_pred}, ignore_index=True)
            dist.append(mse_diff)
            y_true.append(label)
        # else:
        #     print(f"[跳过测试] 当前图像 {img_name} 是参考集图像，writer: {writer}")
        #     print(f"参考集包含该图片: {img_name in ref_img_set}")

    print(f">> Total nos of tested samples: {len(dist)}")
    with open('DEBUG.txt', 'w', encoding='utf-8') as f:
        for t in range(75):
            f.write(f"Sample {t}: Distance = {dist[t]:.4f}, Label = {y_true[t]}\n")
            f.write(f"该sample是图片 {identify_writer[t]} compare to the writer's reference set\n")
            f.write('----------------------------------------\n')

    plt.hist([np.array(dist)[np.array(y_true)==1], 
          np.array(dist)[np.array(y_true)==0]], 
         label=['Genuine', 'Forgery'], bins=50, alpha=0.7)
    plt.xlabel("Mean Distance to Reference")
    plt.ylabel("Count")
    plt.title("Distribution of Distances")
    # 保存下图片
    plt.legend()
    plt.savefig(f"./111.jpg", dpi=300)

    metrics, thresh_optimal = compute_accuracy_roc(np.array(dist), np.array(y_true), step=args.stepsize)

    print("Metrics obtained: \n" + '-'*50)
    print(f"Acc: {metrics['best_acc'] * 100 :.4f} %")
    print(f"FAR: {metrics['best_far'] * 100 :.4f} %")
    print(f"FRR: {metrics['best_frr'] * 100 :.4f} %")
    print('-'*50)

    if args.roc is True:
        if args.roc_name is None:
            args.roc_name = f"{os.path.basename(args.dataset)}_{args.eval_type}_{os.path.basename(args.model_path)}"
            with open(f"logs/{args.roc_name}.json", "w") as outfile:
                json.dump(metrics, outfile)
        plot_roc(metrics['tpr_arr'], metrics['far_arr'], fname=args.roc_name)

