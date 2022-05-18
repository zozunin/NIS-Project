import os, time, sys, csv, io, time
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from scipy.spatial import distance
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms as trn
import torch.functional as F
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)

! unzip '/content/gdrive/MyDrive/ColabNotebooks/new_data.zip'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def read_tags(folder_dir, filename):
    threshold=0

    read_data = []
    word_counter = {}
    file_path   = filename
    with open(file_path, 'r', encoding="utf-8") as f:
        tmp = f.readlines()
        for content in tmp:
            content = content.strip('\n').strip() 
            test = content.strip().split()
            if len(test) > 1:
                read_data.append(test)
                for ele in test:
                    if ele not in word_counter.keys():
                        word_counter[ele] = 1
                    else:
                        word_counter[ele] += 1
    corpus = [k for k,v in word_counter.items() if v > threshold]
    word2idx = {k:v for v,k in enumerate(corpus, 1)}
    word2idx['UNK'] = 0
    idx2word = {v:k for v,k in enumerate(corpus, 1)}
    idx2word[0] = 'UNK'
    corpus = corpus+['UNK']
    return read_data, word_counter, corpus, word2idx, idx2word

class data_reader(Dataset):

    def __init__(self, folder_dir, word2idx=None, word_vec_dict=None, train=True):
        self.train = train
        self.img_centre_crop = img_centre_crop()

        self.folder_dir = folder_dir
        self.data = pd.read_csv('new_data_list.txt', header=None).values
        self.tag = pd.read_csv('new_tag_list.txt', header=None).values
        self.total_num = len(self.data)
        self.rdn_idx = np.arange(self.total_num)
        np.random.shuffle(self.rdn_idx)
        self.data = self.data[self.rdn_idx]
        self.tag = self.tag[self.rdn_idx]
        
        self.x = self.data[:-int(0.2*self.total_num)]
        self.y = self.tag[:-int(0.2*self.total_num)] # все теги в строчку
        self.vx = self.data[-int(0.2*self.total_num):]
        self.vy = self.tag[-int(0.2*self.total_num):]

        self.word2idx = word2idx
        self.word_vec_dict = word_vec_dict

    def __len__(self):
        return(len(self.x) if self.train else len(self.vx))
    
    def __getitem__(self, idx):
        tmp_x = self.x[idx] if self.train else self.vx[idx]
        tmp_y = self.y[idx] if self.train else self.vy[idx]
        
        hashtag = tmp_y[0].strip()
        category = tmp_x[0].split('\\')[1]
        
        category = category if category not in self.word2idx.keys() else category
        widx = self.word2idx[category]
        category_vec = self.word_vec_dict[widx]
        
        tmp_x = tmp_x[0].replace('\\', '/')
        tmp_x = tmp_x.replace('instagram_dataset/', '')
        tmp_path = os.path.join(self.folder_dir ,tmp_x)
        img = Image.open(tmp_path).convert('RGB')

        input_img = self.img_centre_crop(img)

        return(input_img, (category_vec, hashtag))

"""### word2vec part"""

def convert2pair(data, corpus, word2idx):
    skip_window = 3
    idx_data = data.copy()
    for i, row in enumerate(idx_data):
        for i_ele, ele in enumerate(row):
            if ele not in word2idx.keys():
                idx_data[i][i_ele] = 0
            else:
                idx_data[i][i_ele] = word2idx[ele]
    
    training_set = []
    for i, row in enumerate(idx_data):
        for i_ele, ele in enumerate(row):
            for slide in range(-skip_window, skip_window+1):
                if i_ele+slide < 0 or i_ele+slide >= len(row) or slide == 0:
                    continue
                else:
                    training_set.append([ele, row[i_ele+slide]])
    return training_set

class word2vec_dataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return(len(self.data))
    def __getitem__(self, idx):
        return (self.data[idx][0], self.data[idx][1])

class word2vec(nn.Module):
    def __init__(self, vol_size):
        super(word2vec, self).__init__()
        self.embed = 500
        self.vol = vol_size
        self.u_embeddings = nn.Embedding(self.vol, self.embed)
        self.linear = nn.Linear(self.embed, self.vol)
        
    def forward(self, x):
        embed_vec = self.u_embeddings(x)
        #x = torch.matmul(x, self.inver_mat)
        x = self.linear(embed_vec)
        x = torch.nn.functional.log_softmax(x, dim=1)
        return x, embed_vec

folder_dir = r"/content/gdrive/My Drive/ColabNotebooks/HARRISON_RN"
filename = "full_tag_list.txt"

mode = 'train'
batch_size = 64
epochs = 200
lr = 0.0001
    
data, word_counter, corpus, word2idx, idx2word = read_tags(folder_dir, filename)
wordpair = convert2pair(data, corpus, word2idx)

training_data = DataLoader(word2vec_dataset(wordpair), batch_size=batch_size, shuffle=True)

word2vec_model = word2vec(len(corpus)).to(device)
optimizer = optim.AdamW(word2vec_model.parameters(), lr=lr)
criterion = nn.NLLLoss()
    
for epoch in range(epochs):
    total_loss = 0
    print("This is epoch: {:4d}".format(epoch))
    est = time.time()
    for i,(x,y) in enumerate(training_data, 1):
        x = x.to(device)
        y = y.to(device)
        word2vec_model.zero_grad()
        y_pred, _ = word2vec_model(x)
        loss = criterion(y_pred.view(-1, len(corpus)), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i != len(training_data):
            print("Batch: {:6d}/{:6d} loss: {:06.4f}".format(i, len(training_data), total_loss/i), end='\r')
        else:
            eft = time.time()
            print(' '*40, end='\r')
            print("Batch: {:6d}/{:6d} loss: {:06.4f} time: {:04.2f}s".format(i, i, total_loss/i, eft-est))
        
    print("--- This is test part ---")
    
    word2vec_model.eval()
    with torch.no_grad():
        total = np.arange(len(corpus)).reshape((len(corpus),1)) 
        tx = np.random.randint(len(corpus), size=1)
        torch_total = torch.from_numpy(total).to(device)
        _, embed_total = word2vec_model(torch_total)
        embed_total = embed_total.reshape((len(corpus), -1))
            
        embed_total_numpy = embed_total.cpu().numpy()
        dists = embed_total_numpy[tx[0]].dot(embed_total_numpy.transpose())
        knn_word_ind = (-dists).argsort()[0:10]
        print("Top@10 close to {} are: ".format(idx2word[tx[0]]), end=" ")
        for i in knn_word_ind:
            print(idx2word[i], end=" ")
        print()

    with torch.no_grad():
        idx2wordvec = {}
        total = torch.from_numpy(np.arange(len(corpus)).reshape(-1,1)).to(device)
        _, embed_total = word2vec_model(total)
        embed_total = embed_total.reshape((len(corpus), -1))
        embed_total_numpy = embed_total.cpu().numpy()
        np.savez(r'/content/gdrive/My Drive/ColabNotebooks/wordvec_2', wordvec=embed_total_numpy)

"""### image processing"""

def img_centre_crop():
    centre_crop = trn.Compose([
            trn.Resize((256,256)),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return centre_crop


class img2vec(nn.Module):
    def __init__(self):
        super(img2vec, self).__init__()
        self.linear1 = nn.Linear(2048, 1024)
        self.linear2 = nn.Linear(1024, 500) 
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

def img_extractor():
    resnet50 = models.resnet50(num_classes=1000, pretrained=True)
    modules = list(resnet50.children())[:-1]
    resnet50 = nn.Sequential(*modules)
    return resnet50

def cos_cdist(matrix, v):
   return distance.cdist(matrix, v, 'cosine').T


def top_k(y_pred, vec_matrix, k):
    tmp = cos_cdist(vec_matrix, y_pred)
    total_top_k = np.argsort(tmp)[:, :k]
    return total_top_k

def f1_score(top_k_result, hashtag_y, idx2word):
    gt_hashtag, to_word = [], []
    tp, total_l = 0, 0
    for ele in hashtag_y:
        ele = ele.strip().split()
        gt_hashtag.append(ele)
    for row in range(len(top_k_result)):
        tmp = []
        for ele in top_k_result[row]:
            tmp += [idx2word[ele]]
        to_word.append(tmp)
    for row_idx in range(len(to_word)):
        tp += len(set(to_word[row_idx])&set(gt_hashtag[row_idx]))
        total_l += len(gt_hashtag[row_idx])
    return tp, total_l

"""### main training"""

def training(img_extractor_model, img2vec, folder_dir, word2idx, word_vec_dic, vec_matrix):
    
    epochs = 100
    threshold = 4
    batch_size = 64
    lr = 1e-4

    save = '/content/gdrive/My Drive/inst_proj'
    
        tr_img = data_reader(folder_dir, word2idx, word_vec_dict)
    te_img = data_reader(folder_dir, word2idx, word_vec_dict, train=False)
    tr = DataLoader(tr_img, batch_size=batch_size, shuffle=False)
    te = DataLoader(te_img, batch_size=batch_size, shuffle=False)

    optimizer_ex = optim.Adam(img_extractor_model.parameters(), lr=lr)
    optimizer_img = optim.Adam(img2vec.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val, best_tr = 0, 0
    for epoch in range(epochs):
        print('Epoch {:4d}'.format(epoch+1))
        total_loss, all_tp, all_total = 0, 0, 0
        img_extractor_model.train()
        img2vec.train()
        st = time.time()
        for i,(x,y) in enumerate(tr, 1):
            print(f"Ongoing batch {i}")
            cate_y, hashtag_y = y
            x = x.to(device)
            cate_y = cate_y.to(device)
            
            img_extractor_model.zero_grad()
            img2vec.zero_grad()

            img_feat = img_extractor_model(x)
            y_pred = img2vec(img_feat.squeeze())
            loss = criterion(y_pred, cate_y)

            loss.backward()
            optimizer_ex.step()
            optimizer_img.step()
            total_loss += loss.item()
            
            ### find top@k of y_pred and evalution ###
            y_pred_find = y_pred.detach().cpu()
            top_k_result = top_k(y_pred_find, vec_matrix, 10)
            tp, total_l = f1_score(top_k_result, hashtag_y, idx2word)
            all_tp += tp
            all_total += total_l
            recall_rate = all_tp/all_total *100 
            print("Batch: {:4d}/{:4d} loss: {:06.4f}, recall: {:04.2f}%".format(i, len(tr), total_loss/i, recall_rate),
                  end=' '*5+'\r' if i != len(tr) else '\n')
        print('total time: {:>5.2f}s'.format(time.time()-st))
        if recall_rate > best_tr:
            best_tr = recall_rate
            print('saving best training recall model...')
            tr_path = os.path.join(save, 'best_tr')
            torch.save(img_extractor_model.state_dict(), tr_path+f'/_img_extractor.pth')
            torch.save(img2vec.state_dict(), tr_path+f'/_img2vec.pth')

        with torch.no_grad():
            vall_tp, vall_total = 0, 0
            img_extractor_model.eval()
            img2vec.eval()
            for i,(vx,vy) in enumerate(te, 1):
                _, val_hashtag_y = vy
                vx = vx.to(device)

                vimg_feat = img_extractor_model(vx)
                vy_pred = img2vec(vimg_feat.squeeze()).detach().cpu()
                vtop_k_result = top_k(vy_pred, vec_matrix, 10)
                vtp, vtotal_l = f1_score(vtop_k_result, val_hashtag_y, idx2word)
                vall_tp += vtp
                vall_total += vtotal_l
                vrecall_rate = vall_tp/vall_total *100
                print("Testing set recall rate: {:04.2f}% vall_tp: {} vall_total: {}".format(vrecall_rate, vall_tp, vall_total)
                      , end=' '*5+'\r' if i != len(te) else '\n')
            if vrecall_rate > best_val:
                best_val = vrecall_rate
                print('saving best validation recall model...')
                val_path = os.path.join(save, 'best_val')
                torch.save(img_extractor_model.state_dict(), val_path+'/_img_extractor.pth')
                torch.save(img2vec.state_dict(), val_path+'/_img2vec.pth')

def testing(imageID, img_extractor_model, img2vec, word2idx, word_vec_dict, vec_matrix):

    fix_path = '/content/gdrive/My Drive/ColabNotebooks/test'
    crop = img_centre_crop()
    
    tmp_path = os.path.join(fix_path, imageID)
    img = Image.open(tmp_path).convert('RGB')
    input_img = crop(img).unsqueeze(0).to(device)

    tmp = []
    tmp += [imageID]
    with torch.no_grad():
        img_feat = img_extractor_model(input_img)
        pred = img2vec(img_feat.squeeze().unsqueeze(0)).detach().cpu()
        top_k_result = top_k(pred, vec_matrix, 10)[0]
        for ele in top_k_result:
            tmp += [idx2word[ele]]
        print(tmp)
    return tmp

mode = "test"

start_time = time.time()

# current data dirs
folder_dir = "fixed_data"
filename = "new_tag_list.txt"

# from pre-trained w2v
npz_path = r'/content/gdrive/My Drive/ColabNotebooks/wordvec_2.npz'
folder_dir_ = r"/content/gdrive/My Drive/ColabNotebooks/HARRISON_RN"
filename_ = "full_tag_list.txt"
temp = os.path.join(folder_dir_, filename_)

# for test part
# imgID = 'image_41.jpg'
m_path = r'/content/gdrive/My Drive/inst_proj'

vec_matrix = np.load(npz_path)['wordvec']
word_vec_dict = {idx:vec for idx,vec in enumerate(vec_matrix)}
_, _, corpus, word2idx, idx2word = read_tags(folder_dir, temp)

img_extractor_model = img_extractor().to(device)
img2vec = img2vec().to(device)

if mode == "train":
        training(img_extractor_model, img2vec, folder_dir, word2idx, word_vec_dict, vec_matrix)

else:
    img_extractor_model.load_state_dict(torch.load(m_path+'/best_tr/_img_extractor.pth'))
    img2vec.load_state_dict(torch.load(m_path+'/best_tr/_img2vec.pth'))

    img_extractor_model.eval()
    img2vec.eval()
    result = testing(imgID, img_extractor_model, img2vec, word2idx, word_vec_dict, vec_matrix)
