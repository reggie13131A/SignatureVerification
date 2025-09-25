import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import torch
from torchvision.utils import save_image
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC as SVM
# 这个是从dataset中调用了全部了
from dataset import *
from model import *
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(1)
random.seed(1)
torch.manual_seed(1)

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
    for d in np.arange(dmin, dmax + step, step):
        idx1 = predictions.ravel() <= d     # pred = 1
        idx2 = predictions.ravel() > d      # pred = 0

        tpr = float(np.sum(labels[idx1] == 1)) / nsame
        tnr = float(np.sum(labels[idx2] == 0)) / ndiff

        acc = 0.5 * (tpr + tnr)
        
        # print(f"Threshold = {d} | Accuracy = {acc:.4f}")

        if acc > max_acc:
            max_acc = acc
            d_optimal = d
            
            # FRR, FAR metrics
            min_frr = float(np.sum(labels[idx2] == 1)) / nsame
            min_far = float(np.sum(labels[idx1] == 0)) / ndiff
            
    metrics = {"best_acc" : max_acc, "best_frr" : min_frr, "best_far" : min_far}
    return metrics, d_optimal


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser('PatchAttnReconstruction | SSL for Writer Identification')
    # parser.add_argument('--base_dir', type=str, default=os.getcwd())
    parser.add_argument('--dataset', type=str, default='ChiSig')
    parser.add_argument('--base_dir', type=str, default='/home/admin-ps/Documents/hatAndMask/SignatureDatasets/')
    parser.add_argument('--saved_models', type=str, default='./saved_models')
    parser.add_argument('--batchsize', type=int, default=1)  # change for Testing; default=16
    parser.add_argument('--print_freq', type=int,default=5)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--Encoder_learning_rate', type=int, default=3e-4)
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--fmap_dims', type=int, default=8)
    parser.add_argument('--ptsz', type=int, default=128)
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--is_pretrained', type=bool, default=False)
    parser.add_argument('--cls', type=str, default='knn')
    parser.add_argument('--model_path', type=str, default="/home/admin-ps/Documents/hatAndMask/signatureVerification/SURDS-SSL-OSV-main/PatchAttnRecons_SSL_OSV/saved_models/Chisig_ChiSig_R=None_SSL.pt")
    parser.add_argument('--stepsize', type=float, default=0.0005)
    args = parser.parse_args()

    print('\n'+'*'*100)

    # 1. get data
    train_loader, test_loader = get_dataloader(args)
    
    # 2. load model
    MODEL_PATH = args.model_path

    checkpoint = torch.load(MODEL_PATH)
    print(f"Loading model from: {MODEL_PATH} | Epochs trained: {checkpoint['epochs']}")
    # SSL类下的model，先创建一个实例
    model = SSL_Model(args)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    # 3. feature extraction from train and test
    features, labels, writer_ids, img_names = [], [], [], []
    mean_AP, mean_AN = 0.0, 0.0

    with torch.no_grad():
        # test laoder是pytorch之下的一个类，经历了getPatch、resize等操作
        for batch in test_loader:
            # 裁剪之后
            batch['image'] = batch['image'].to(device)      # [N,3,256,256] 
            # 打碎之后
            batch['patches'] = batch['patches'].to(device)  # [N,16,3,64,64]
            patch_attn_feats = []
            # 这是一整张图片进去，encoder是resnet18
            image_feature = model.encoder(batch['image'], pool=True)
            # 论文中的第二分支
            for patch in batch['patches'][0]: 
                # 一patch一pach的进去
                patch_feature_map = model.encoder(patch.unsqueeze(0), pool=False) # [N,512,1,1]
                # 2D注意力机制接收 img_fea，patch_fea
                _, attn_feature = model.attn_module(image_feature, patch_feature_map) # [N,512]
                # 输出后的confused结果
                patch_attn_feats.append(attn_feature.unsqueeze(1)) # [N,1,512]
            # 将列表中的元素在 dim = 1 cat起来
            patch_attn_feats = torch.cat(patch_attn_feats, dim=1) # [N,16,512]
            patch_attn_feats = patch_attn_feats.permute(0,2,1) # [N,512,16]
            # 可能这个是decoder
            patch_attn = model.avg_pool(patch_attn_feats) # [N,512,1]
            patch_attn = patch_attn.permute(0,2,1).squeeze(1) # [N,512]

            features.append(patch_attn.cpu().numpy())
            labels.append(batch['label'].cpu().numpy())
            # writer_ids.append(int(batch['writer_id'][0]))
            writer_ids.append(batch['writer_id'][0])
            img_names.append(batch['img_name'][0])
        # print(len(features), len(labels), len(writer_ids), len(img_names))

    features = np.array(features)
    labels = np.array(labels)
    writer_ids = np.array(writer_ids)

    # print(features.shape, labels.shape)
    n_samples = features.shape[0] * features.shape[1]
    n_features = features.shape[2]

    features = features.reshape(n_samples, n_features)
    labels = labels.reshape(labels.shape[0])
    # print(features.shape, labels.shape)

    # X = features.copy()
    # y = labels.copy()
    # W_id = writer_ids.copy()
    # X_train, X_test, y_train, y_test, Wid_train, Wid_test = train_test_split(X, y, W_id, test_size=0.3, shuffle=False, random_state=1)

    ### SigNet Evaluation ###
    df = pd.DataFrame(features)
    df['label'] = labels.copy()
    df['writer_id'] = writer_ids.copy()
    df['img_name'] = img_names.copy()

    wrtr_set = set()
    df_ref_writer = pd.DataFrame()

    for i in range(len(df)):
        label = df.iloc[i]['label']    # 当前的样本标签
        writer = df.iloc[i]['writer_id']    # 当前的样本作家
        img_name = df.iloc[i]['img_name']     # 当前的样本图片名

        if writer not in wrtr_set:
            wrtr_set.add(writer)
            # print(f">> Creating reference set for Writer ID: {writer}")
            # create reference set for current writer
            df_ref = df[(df['writer_id']==writer) & (df['label']==1)]
            # assert (len(df_ref) == 24)
            print(f"作家 {writer} 的真实签名数量: {len(df_ref)}")
            # if len(df_ref) < 5:
            #     print(f"Writer {writer} has less than 5 genuine signatures, skipping...")
            #     continue
            assert (len(df_ref) == 5), f"错误：作家 {writer} 只有 {len(df_ref)} 个真实签名"
            df_ref = df_ref[(df_ref['img_name'] != img_name)]
            # print(f"Genuine set excluding current image for writer {writer} is: {len(df_ref)}")
            """
                这里使用3张照片作为参考set
                论文中是24张照片作为参考set
            """
            df_ref = df_ref.sample(3, random_state=0)  
            assert (len(df_ref) == 3)
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

    for i in range(len(df)):    # 遍历所有样本
        feature = np.array(df.iloc[i][0:1536]).flatten() # the feature of current sample which dim = 1536
        label = df.iloc[i]['label']     
        writer = df.iloc[i]['writer_id']     
        # img_name = df.iloc[i]['img_name']    
        img_name = str(df.iloc[i]['img_name']).strip()
        
        if img_name not in ref_img_set:
            identify_writer.append(img_name)
            df_ref = df_ref_writer[(df_ref_writer['writer_id']==writer)]
            assert (len(df_ref) == 3)
            df_ref = df_ref.drop(['label', 'writer_id', 'img_name'], axis=1)
            mean_ref = np.mean(np.array(df_ref, dtype=np.float32), axis=0)
            mse_diff = np.abs(np.mean(np.subtract(feature, mean_ref)))
            # y_pred = 1 if mse_diff <= THRESHOLD else 0
            # preds = preds.append({'img_name' : img_name, 'writer_id' : writer, 'y_true' : label, 'y_pred' : y_pred}, ignore_index=True)
            dist.append(mse_diff)
            y_true.append(label)

    print(f">> Total nos of tested samples: {len(dist)}")
    print(f"identify_writer length: {len(identify_writer)}")
    with open('DEBUG.txt', 'w', encoding='utf-8') as f:
        for t in range(75):
            f.write(f"Sample {t}: Distance = {dist[t]:.4f}, Label = {y_true[t]}\n")
            f.write(f"该sample是图片 {identify_writer[t]} compare to the writer's reference set\n")
            f.write('----------------------------------------\n')

    metrics, thresh_optimal = compute_accuracy_roc(np.array(dist), np.array(y_true), step=args.stepsize)

    print("Metrics obtained: \n" + '-'*50)
    print(f"Acc: {metrics['best_acc'] * 100 :.4f} %")
    print(f"FAR: {metrics['best_far'] * 100 :.4f} %")
    print(f"FRR: {metrics['best_frr'] * 100 :.4f} %")
    print('-'*50)

    ############################################################################################################
    # eval metrics ##
    # y_true, y_pred = np.array(preds['y_true']), np.array(preds['y_pred'])
    # accuracy = np.sum(y_pred == y_true)/y_true.shape[0]
    # print(f'Accuracy using THRESHOLD = {THRESHOLD}: {accuracy:.6f}')

    # trues = 0
    # for i in range(len(preds)):
    #     if preds['y_true'][i] == 1 and preds['y_pred'][i] == 1:
    #         trues += 1
    # print(f"Correctly preds True: {trues}")

    # preds.to_csv("./saved_models/preds.csv", index=False)
    #############################################################################################################


    # 4. train a KNN classifier
    # cls = SVM() # if args.cls.upper() == 'SVM' else KNN()
    # cls.fit(X_train, y_train)
    # y_pred = cls.predict(X_test)
    # accuracy = np.sum(y_pred == y_test)/y_test.shape[0]
    # print(f'Accuracy using SVM classifier: {accuracy:.6f}')


    # # 5. plot tsne
    # tsne = TSNE(n_components=2, perplexity=50, random_state=0)

    # X_train_tsne = tsne.fit_transform(X_train)
    # X_test_tsne = tsne.fit_transform(X_test)

    # plt.figure(figsize=(16,16))
    # sns.scatterplot(
    #             x=X_train_tsne[:,0], y=X_train_tsne[:,1],
    #             # hue=np.array(y_train),            
    #             # palette=sns.color_palette(["Red", "Blue"], 2),
    #             hue=Wid_train,
    #             palette=sns.color_palette("bright", np.unique(Wid_train).shape[0]),
    #             legend="full",
    #             alpha=0.5,
    #         )
    # plt.savefig(f"./saved_models/DualTriplet_train_tsne_SigNet_WriterWise_LR=0.0001.png", dpi=300)

    # plt.clf()

    # plt.figure(figsize=(16,16))
    # sns.scatterplot(
    #             x=X_test_tsne[:,0], y=X_test_tsne[:,1],
    #             # hue=np.array(y_test),  
    #             # palette=sns.color_palette(["Red", "Blue"], 2),          
    #             hue=Wid_test,
    #             palette=sns.color_palette("bright", np.unique(Wid_test).shape[0]),
    #             legend="full",
    #             alpha=0.5,
    #         )
    # plt.savefig(f"./saved_models/DualTriplet_test_tsne_SigNet_WriterWise_LR=0.0001.png", dpi=300)


    # K-medoids
    # cls = KMedoids(n_clusters=30, random_state=0)
    # kmeds = cls.fit(X_test)

    # med_indices = kmeds.medoid_indices_
    # medoid_imgs = list(Wid_test[med_indices])
    # medoid_imgs.sort()
    # medoid_imgs = pd.DataFrame(medoid_imgs)
    # medoid_imgs.to_csv(f"./saved_models/WriterWise_SigNet_Clusters_MedoidImages.csv", index=False)

    # clusters = kmeds.predict(X)
    # plt.figure(figsize=(16,16))
    # sns.scatterplot(
    #     x=X_test_tsne[:,0], y=X_test_tsne[:,1],
    #     hue=np.array(clusters),
    #     palette=sns.color_palette("hls", N),
    #     legend="full",
    #     alpha=0.5,
    # )
    # plt.savefig(f"./saved_figures/{args.algo}_N={N}.jpg", dpi=300)

