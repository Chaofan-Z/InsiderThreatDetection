import os
from sys import version
import torch.utils.data as Data
import numpy as np
import torch
import pickle
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sequential_model import Sequential_model
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from torch import nn

def check_session_embedding(session_embedding, maxLen=30):
    if len(session_embedding) > 30:
        return session_embedding[:30]
    
    for i in range(30 - len(session_embedding)):
        session_embedding.append(zero_emebdding)
    return session_embedding

def load_node_embeddings(embedding_dir):
    # embedding_dir = "../ltc_insider/output/r_part/120211216-embedding.pickle"
    embedding_dir = "/home/zhengchaofan/project/ltc_insider/output/r5.2/5end-embedding.pickle"
    with open(embedding_dir, 'rb') as file:
        embeddings = pickle.load(file)
    return embeddings

def load_data(ourput_data_dir, batch_size=64):
    st = time.time()
    print("Start to load data .. ")
    embeddings = load_node_embeddings("./output/%s/%s/embedding.pickle"%(data_version, version))
    # embeddings = {'test' : zero_emebdding}

    has_embedding = 0
    no_embedding = 0

    all_session_embedding = []
    all_session_label = []
    with open(ourput_data_dir + "sample_session") as file:
        for line in tqdm(file):
            session_embedding = []
            session = line.strip().split('\t')
            for node_id in session:
                if node_id in embeddings:
                    # node_embedding = embeddings[node_id].cpu().detach().numpy().tolist()
                    node_embedding = embeddings[node_id].tolist()
                    # print(node_id)
                    has_embedding += 1
                else :
                    node_embedding = zero_emebdding
                    no_embedding += 1
                session_embedding.append(node_embedding)
            session_embedding = check_session_embedding(session_embedding)
            all_session_embedding.append(session_embedding)
            # break
    print(f"activity has embedding : {has_embedding}, no embedding : {no_embedding}")

    with open(ourput_data_dir + "sample_session_label") as file:
        for line in file:
            all_session_label.append(int(line.strip()))
            # break

    assert len(all_session_embedding) == len(all_session_label)

    print("list to tensor ")
    all_session_embedding = torch.Tensor(all_session_embedding)
    all_session_label = torch.Tensor(all_session_label)
    all_session_label = all_session_label.long()

    # print("data to gpu ", device)
    # all_session_embedding = all_session_embedding.to(device)
    # all_session_label = all_session_label.to(device)

    # 前70%作为训练集，后面作为测试集
    train_rate = 0.7
    split_number =int(len(all_session_embedding)*train_rate)
    train_data = Data.TensorDataset(all_session_embedding[:split_number], all_session_label[:split_number])
    test_data = Data.TensorDataset(all_session_embedding[split_number:], all_session_label[split_number:])
    
    train_data_iter = Data.DataLoader(train_data, batch_size, shuffle=False)
    test_data_iter = Data.DataLoader(test_data, batch_size, shuffle=False)

    print("End to load data, cost ", time.time() - st)
    return train_data, test_data, train_data_iter, test_data_iter

def precesion_recall(confusion):
    metrics = {}
    f1 = 0
    for label in range(len(confusion)):
        if label not in metrics:
            metrics[label] = {}
        metrics[label]["recall"] = confusion[label][label] / sum(confusion[label])
        metrics[label]["precision"] = confusion[label][label] / sum(confusion[:][label])
        if metrics[label]["precision"] + metrics[label]["recall"] == 0:
            metrics[label]["F1"] = 0
        else:
            metrics[label]["F1"] = 2 * metrics[label]["recall"] * metrics[label]["precision"] / (metrics[label]["precision"] + metrics[label]["recall"])
        f1 += metrics[label]["F1"]
    for label in metrics:
        print("label : %s, recall : %.2f, precision : %.2f, F1 : %.2f" % (label, metrics[label]['recall'], metrics[label]['precision'], metrics[label]["F1"]) )
    return f1 / len(confusion)

def plot_confusion_matrix(cm, labels_name, title):
    print("Confusion matrix graph")
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')    
    plt.xlabel('Predicted label')
    plt.savefig("./output/confusion.jpg")

def test(model, test_data_iter):
    pred_label = torch.Tensor([]).to(device)
    pred_score = []
    true_label = torch.Tensor([]).to(device)
    model.eval()
    with torch.no_grad():
        for X, y in tqdm(test_data_iter):
            X = X.to(device)
            y = y.to(device)
            output = model(X)
            # output batch * 5
            pred_score_ = output.cpu().numpy()
            pred_score.append(pred_score_ / np.sum(pred_score_, axis=1, keepdims=True))
            output_label = torch.argmax(output, -1)
            pred_label = torch.cat([pred_label, output_label])
            true_label = torch.cat([true_label, y])
            # break

    pred = np.array(pred_label.cpu())
    true = np.array(true_label.cpu())
    confusion = confusion_matrix(true, pred)

    print("Confusion matrix")
    print(confusion)

    # plot_confusion_matrix(confusion, labels_name=[0,1,2,3,4], title="Threat detection in r5.2")
    marco_f1 = precesion_recall(confusion)

    onehot_encoder = OneHotEncoder(sparse=False)
    true = true.reshape((-1, 1))
    yPred_encode = pred.reshape((-1, 1))
    onehot_encoder.fit(true)
    # numpy.concatenate
    yTest_encode = onehot_encoder.transform(true)
    yPred_encode = onehot_encoder.transform(yPred_encode)
    print('auc: ', roc_auc_score(yTest_encode, yPred_encode, multi_class='ovr'))

    # auc = roc_auc_score(true, np.array(pred_score), multi_class='ovr')
    # print("-----AUC : ", auc)

    return marco_f1
if __name__ == '__main__' :

    torch.cuda.set_device(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_version = "r5.2"
    version = "5"
    # version = "half"

    # data_version = "r_part"
    # version = "1"

    ourput_data_dir = "./output/%s/session_data/"%(data_version)
    # ourput_data_dir = "./output/%s/session_data/backup/r5.2_origin/"%(data_version)
    model_save_path = "./output/%s/%s/sequential_model/"%(data_version, version)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    epochs = 1000
    batch_size = 4096
    zero_emebdding = [0] * 128

    train_data, test_data, train_data_iter, test_data_iter = load_data(ourput_data_dir, batch_size=batch_size)
    print(device)
    model = Sequential_model(seq_len=30, input_size=128, hidden_size=256, output_size=5, batch_first=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    # criterion = nn.NLLLoss().to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1,40,20,40,20])).to(device)

    model_file_path = model_save_path+"model_sample.pth"
    if os.path.exists(model_file_path):
        print("Load model from : ", model_file_path)
        model.load_state_dict(torch.load(model_file_path))

    cost_time = []
    test_marco_f1_best = 0
    print("Start to train")
    for epoch in range(epochs):
        # for i in range(len(input)):
        st_time = time.time()
        all_loss = 0
        model.train()
        for X, y in tqdm(train_data_iter):
            X = X.to(device)
            y = y.to(device)
            output = model(X)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_loss += loss

        ed_time = time.time()
        cost_time.append(ed_time - st_time)
        print("====epoch %s, loss : %.5f===="%(epoch + 1, all_loss.item()))
        if (epoch + 1) % 2 == 0:
            marco_f1 = test(model, test_data_iter)
            if test_marco_f1_best < marco_f1:
                test_marco_f1_best = marco_f1
                print("best_f1 : ", marco_f1)
                print("save model to ", model_save_path + f"model_sample.pth")
                torch.save(model.cpu().state_dict(), model_save_path + f"model_sample.pth")
                model = model.to(device)
            print("=" * 20)
            