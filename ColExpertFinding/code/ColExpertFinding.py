import numpy as np
import tensorflow as tf
from networkx import to_numpy_matrix
import networkx as nx
import datetime
import sys
import os
import pickle

try:
    import ujson as json
except:
    import json
import math
from scipy.linalg import fractional_matrix_power
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import math

class QRouting:    
    
    def  __init__(self,data):        
        self.dataset=data
        self.node_size=self.loadG()
        
    def init_model(self):
                
        #regression layer
        self.regindim=2*11
        self.W1=QRouting.weight_variable((self.regindim,8))
        #self.W2=EndCold.weight_variable((self.W1.shape[1],8))
        #self.W3=EndCold.weight_variable((self.W2.shape[1],16))
        self.W4 = QRouting.weight_variable2(self.W1.shape[1])
        #self.W4 = EndCold.weight_variable2(4*self.GCNW_2.shape[1])
        self.b = tf.Variable(random.uniform(0, 1))
        self.inputs=[]
        self.outputs=[]    
        
        self.n_bins=11 #number of kernels
        self.wordembedding_size=300        
        self.lamb = 0.5

        self.mus = QRouting.kernal_mus(self.n_bins, use_exact=True)
        self.sigmas = QRouting.kernel_sigmas(self.n_bins, self.lamb, use_exact=True)
        
        self.wordembeddings = tf.Variable(tf.random.uniform([self.vocab_size+1, self.wordembedding_size], -1.0, 1.0,dtype=tf.float32),dtype=tf.float32)
        
        self.nodeembedding_size=128
        self.nodeembeddings = tf.Variable(tf.random.uniform([self.node_size, self.nodeembedding_size], -1.0, 1.0,dtype=tf.float32),dtype=tf.float32)
       
        
    def weight_variable(shape):
        tmp = np.sqrt(6.0) / np.sqrt(shape[0]+shape[1])
        initial = tf.random.uniform(shape, minval=-tmp, maxval=tmp)
        return tf.Variable(initial,dtype=tf.float32)
    
    def weight_variable2(shape):
        tmp = np.sqrt(6.0) / np.sqrt(shape)
        initial = tf.random.uniform([shape,1], minval=-tmp, maxval=tmp)
        return tf.Variable(initial,dtype=tf.float32)
    
    def loadG(self):        
        fin=open(self.dataset+"/CQG_proporties.txt","r")
        N=int(fin.readline().strip().split(" ")[2])
        fin.close()
        self.neighbors=[]
        for i in range(N):
            self.neighbors.append([i])
        fin=open(self.dataset+"/CQG.txt","r")
        line=fin.readline().strip()
        while line:
            d=line.split(" ")
            s=int(d[0])
            e=int(d[1])
            self.neighbors[s].append(e)
            self.neighbors[e].append(s)
            line=fin.readline().strip()
        fin.close()  
        return N
   
    def load_traindata(self,qlen,alen):
        """load tain data"""
        self.train_data=[]
        self.train_label=[]
        self.train_data_neighbors=[]
        
        INPUT=self.dataset+"/train_data.txt"
        fin_train=open(INPUT)
        INPUT2=self.dataset+"/train_labels.txt"
        fin_label=open(INPUT2)
        train=fin_train.readline().strip()
        label=fin_label.readline().strip()
        while train:
            data=train.split(" ")
            lst=[]
            for d in data:
                lst.append(int(d)) 
            qid=lst[0]
            answererid=lst[2]
            qneighboirs=self.neighbors[qid].copy()
            qneighboirs.remove(answererid)
            eneigbors=self.neighbors[answererid].copy()
            eneigbors.remove(qid)
            self.train_data_neighbors.append( [qneighboirs,eneigbors]) 
            
            #print(self.train_data_neighbors)
            #sys.exit(0)
            self.train_data.append(lst)
            train=fin_train.readline().strip()
            datal=float(label)
            self.train_label.append(datal)
            label=fin_label.readline().strip()
        fin_train.close()
        fin_label.close()
        self.train_data=np.array(self.train_data)
        self.train_data_neighbors=np.array(self.train_data_neighbors)
        #self.train_label=np.array(self.train_label)
        
        #add nagetive samples
        INPUT=self.dataset+"/CQG_proporties.txt"        
        pfile=open(INPUT)
        line=pfile.readline()
        N=int(line.split(" ")[2]) # number of nodes in the CQA network graph N=|Qestions|+|Askers|+|Answerers|+|tags|
        line=pfile.readline()
        qnum=int(line.split(" ")[2])     
        user_id_map={}
        INPUT3=self.dataset+"/user_id_map.txt"
        fin=open(INPUT3, "r",encoding="utf8")
        line=fin.readline().strip()
        while line:            
            e=line.split(" ")
            uname=" ".join(e[1:])            
            uname=int(uname.strip())
            user_id_map[uname]=qnum+int(e[0])            
            line=fin.readline().strip()
        fin.close() 
        
        
        answerers=[]
        INPUT=self.dataset+"/user_answers.txt"
        with open( INPUT, "r") as fin:                
            for line in fin:
                d = line.strip().split(" ")        
                answerers.append(int(d[0]))
        new_data=[]
        new_data_neighbors=[]
        ids=np.array([self.train_data[i][0] for i in range(self.train_data.shape[0])])
        
        
        for i in set(ids): 
            #print(i)
            ind=np.where(ids==i)
            answerer_posids=[ a[2] for a in self.train_data[ind]]
            #print(answerer_posids)
            qaetinfo=self.train_data[ind][0].copy()
            #print(qaetinfo)
            qaetinfo[3]=-1
            for kk in ind[0]:
                
                neid=user_id_map[random.choice(answerers)]
                while neid in answerer_posids:
                    neid=user_id_map[random.choice(answerers)]
                #qaetinfo[2]=neid
                p1=qaetinfo[0:2].copy()
                p1.append(neid)
                p1.extend(qaetinfo[3:])
                
                new_data.append([self.train_data[kk] , p1 ])
                
                qid=p1[0]
                answererid=p1[2]
                qneighboirs=self.neighbors[qid].copy()
                
                eneigbors=self.neighbors[answererid].copy()
                          
                new_data_neighbors.append([self.train_data_neighbors[kk] ,[qneighboirs,eneigbors]])
            
        self.train_data=np.array(new_data)
        self.train_data_neighbors=np.array(new_data_neighbors)
        #print("ok:")
        #print(self.train_data[0])
        #print(self.train_data_neighbors[0])
        self.train_label=np.array(self.train_label)
        #print(self.train_data[-10:])
        #sys.exit(0)
        #end nagetive
        
        #print(self.train_label[:20])
        #sys.exit(0)       
        #shuffle
        ind_new=[i for i in range(len(self.train_data))]
        np.random.shuffle(ind_new)
        self.train_data=self.train_data[ind_new,]        
        self.train_data_neighbors=self.train_data_neighbors[ind_new,]
        # load q and answer textself.train_data_neighbors
        
        self.qatext=[]
        answers={}
        qtitle={}
        qcontent={}
        vocab=[]
        
        INPUT=self.dataset+"/vocab.txt"
        fin=open( INPUT, "r")
        line=fin.readline()
        line=fin.readline().strip()
        while line:
            v = line.split(" ")        
            vocab.append(v[0])
            line=fin.readline().strip()
        
        INPUT=self.dataset+"/A_content_nsw.txt"
        with open( INPUT, "r") as fin:                
            for line in fin:
                d = line.strip().split(" ")        
                answers[int(d[0])]=d[1:]
        
        INPUT=self.dataset+"/Q_content_nsw.txt"
        with open( INPUT, "r") as fin:                
            for line in fin:
                d = line.strip().split(" ")        
                qcontent[int(d[0])]=d[1:]
        
        INPUT=self.dataset+"/Q_title_nsw.txt"
        with open( INPUT, "r") as fin:                
            for line in fin:
                d = line.strip().split(" ")        
                qtitle[int(d[0])]=d[1:]        
        
        Q_id_map={}
        INPUT2=self.dataset+"/Q_id_map.txt"
        ids=np.loadtxt(INPUT2, dtype=int)
        for e in ids:
            Q_id_map[int(e[0])]=int(e[1])
        
        u_answers={}
        INPUT=self.dataset+"/user_answers.txt"
        with open( INPUT, "r") as fin:                
            for line in fin:
                d = line.strip().split(" ")        
                u_answers[user_id_map[int(d[0])]]=d[1::2]
        
        self.max_q_len=qlen
        self.max_d_len=alen
        self.vocab_size=len(vocab)
        
        delindx=0
        delindexes=[]
        for td in self.train_data:
            #print(td)
            qid=Q_id_map[td[0][0]]
            #print(qid)
            
            aid=td[0][3] 
            #print(aid)
            qtext=qtitle[qid].copy()
            qtext.extend(qcontent[qid])            
            qtext=qtext[:self.max_q_len]
            #print(qtext)
            qt=[]
            for wr in qtext:
                qt.append(vocab.index(wr)+1)
            padzeros=self.max_q_len-len(qt)
            #for zz in range(padzeros):
                 #qt.append(0)
            if aid!=-1:        
                atext=answers[aid]
                atext=atext[:self.max_d_len]
                #print(atext)
                at=[]
                for wr in atext:
                    if wr in vocab:
                        at.append(vocab.index(wr)+1)
                    else:
                        print(str(wr)+" not in  vocab" )

                padzeros=self.max_d_len-len(at)
                #for zz in range(padzeros):
                #     at.append(0)
            else:
                e=td[0][2]
                etext1=[]
                for aid in u_answers[int(e)]:
                        #print(aid)
                        etext1.extend(answers[int(aid)][:100])
                        #etext1.extend(answers[int(aid)])
                    
                    #print("inter")
                    #print(inter)
                etext=etext1
                    #etext=etext1
                if len(etext1)>self.max_d_len:                         
                        etext=random.sample(etext1,self.max_d_len)
                        
                
                #print(etext)
                etext2=[]
                for ii in range(len(etext)):
                    if etext[ii] in vocab:
                        etext2.append(vocab.index(etext[ii])+1)
                    else:
                        print(str(etext[ii])+" not in  vocab" )
                at=etext2.copy()    
            #self.qatext.append([qt,at])
            if len(qt)==0 or len(at)==0:
                delindexes.append(delindx)
            else: 
                pos_txt=[qt,at]
                
            #print(td[1])
            aid=td[1][3]
            if aid!=-1:        
                atext=answers[aid]
                atext=atext[:self.max_d_len]
                #print(atext)
                at=[]
                for wr in atext:
                    if wr in vocab:
                        at.append(vocab.index(wr)+1)
                    else:
                        print(str(wr)+" not in  vocab" )

                padzeros=self.max_d_len-len(at)
                #for zz in range(padzeros):
                #     at.append(0)
            else:
                e=td[1][2]
                etext1=[]
                for aid in u_answers[int(e)]:
                        #print(aid)
                        etext1.extend(answers[int(aid)][:100])
                        #etext1.extend(answers[int(aid)])
                    
                    #print("inter")
                    #print(inter)
                etext=etext1
                    #etext=etext1
                if len(etext1)>self.max_d_len:                         
                        etext=random.sample(etext1,self.max_d_len)
                        
                
                #print(etext)
                etext2=[]
                for ii in range(len(etext)):
                    if etext[ii] in vocab:
                        etext2.append(vocab.index(etext[ii])+1)
                    else:
                        print(str(etext[ii])+" not in  vocab" )
                at=etext2.copy()    
            #self.qatext.append([qt,at])
            if len(qt)==0 or len(at)==0 and delindx not in delindexes:
                delindexes.append(delindx)
            else: 
                neg_txt=[qt,at]
            
            
            delindx+=1 
            self.qatext.append([pos_txt,neg_txt])
            #print( self.qatext)
            #sys.exit(0)
            
        
        self.qatext=np.array(self.qatext) 
#         print(self.qatext[:3])
#         print(self.train_data[:3])
#         print(delindexes)
        if len(delindexes)!=0: #remove q with no answer
            self.train_data=np.delete(self.train_data,delindexes)
            self.train_label=np.delete(self.train_label, delindexes)
            #self.qatext=np.delete(self.qatext,delindexes)
#         print(self.qatext[:3])  
#         print(self.train_data[:3])

        self.val_data,self.val_data_neighbors,self.val_data_text=self.load_test()
        
       
        
        
    def load_test(self):
        """load test data for validation"""        
        INPUT=self.dataset+"/test_data.txt"        
        fin_test=open(INPUT)        
        test=fin_test.readline().strip()
        test_data=[]
        
        while test:
            data=test.split(";")
            lst=[]
            for d in data[0].split(" "):
                lst.append(int(d)) 
            
            alst=[]
            
            for d in data[1].split(" ")[0::3]:
                alst.append(int(d))
            
            anlst=[]
            for d in data[1].split(" ")[1::3]:
                anlst.append(int(d))
            scoresanlst=[]
            for d in data[1].split(" ")[2::3]:
                scoresanlst.append(int(d))
            neg_e=[]
            pos_e=[]
            p_anlst=[]
            for iii in range(len(anlst)):
                if anlst[iii]==-1:
                    neg_e.append(alst[iii])
                else:
                    pos_e.append(alst[iii])
                    p_anlst.append(anlst[iii])
                    
            test_data.append([lst,alst,anlst,scoresanlst,pos_e,neg_e,p_anlst])
            
            test=fin_test.readline().strip()
        fin_test.close()       
        INPUT=self.dataset+"/CQG_proporties.txt"        
        pfile=open(INPUT)
        line=pfile.readline()
        N=int(line.split(" ")[2]) # number of nodes in the CQA network graph N=|Qestions|+|Askers|+|Answerers|+|tags|
        line=pfile.readline()
        qnum=int(line.split(" ")[2])     
        user_id_map={}
        INPUT3=self.dataset+"/user_id_map.txt"
        fin=open(INPUT3, "r",encoding="utf8")
        line=fin.readline().strip()
        while line:            
            e=line.split(" ")
            uname=" ".join(e[1:])            
            uname=uname.strip()
            user_id_map[uname]=qnum+int(e[0])            
            line=fin.readline().strip()
        fin.close()    
        answers={}
        qtitle={}
        qcontent={}
        vocab=[]
        INPUT=self.dataset+"/vocab.txt"
        fin=open( INPUT, "r")
        line=fin.readline()
        line=fin.readline().strip()
        while line:
            v = line.split(" ")        
            vocab.append(v[0])
            line=fin.readline().strip()
        
        INPUT=self.dataset+"/A_content_nsw.txt"
        with open( INPUT, "r") as fin:                
            for line in fin:
                d = line.strip().split(" ")        
                answers[int(d[0])]=d[1:]
        
        INPUT=self.dataset+"/Q_content_nsw.txt"
        with open( INPUT, "r") as fin:                
            for line in fin:
                d = line.strip().split(" ")        
                qcontent[int(d[0])]=d[1:]
        
        INPUT=self.dataset+"/Q_title_nsw.txt"
        with open( INPUT, "r") as fin:                
            for line in fin:
                d = line.strip().split(" ")        
                qtitle[int(d[0])]=d[1:] 
        
        Q_id_map_to_original={}
        INPUT2=self.dataset+"/Q_id_map.txt"
        ids=np.loadtxt(INPUT2, dtype=int)
        for e in ids:
            Q_id_map_to_original[int(e[0])]=int(e[1])
            
        max_q_len=20
        max_d_len=100
        u_answers={}
        INPUT=self.dataset+"/user_answers.txt"
        with open( INPUT, "r") as fin:                
            for line in fin:
                d = line.strip().split(" ")        
                u_answers[int(d[0])]=d[1::2]
                
        
        batch_size=1     
        #results=[]        
        iii=0
        val_data=[]
        val_labels=[]
        val_qatext=[]
        val_data_neighbors=[]
        for tq in test_data:
            #print(iii)
            iii=iii+1
            #print("test q:")
            #print(tq)            
           
            ids=tq[1]  
            
            pos_e=tq[4]
            neg_e=tq[5]
            pos_e_answers=tq[6]
            
            answerids=tq[2]
            scoresanlst=tq[3]
            #print("experts:")      
            #print(ids)
            inputs=[]
            inputtext=[]
            
            qtext=[]
            qid=Q_id_map_to_original[int(tq[0][0])]
            qtext1=qtitle[qid].copy()
            qtext1.extend(qcontent[qid])
            qtext1=qtext1[:20]
            qtext=qtext1.copy()
            #print(qtext)
            for i in range(len(qtext)):
                qtext[i]=vocab.index(qtext[i])+1
            
            #if len(qtext)<max_q_len:                
            #        for i in range(max_q_len-len(qtext)):
                        #qtext.append(0)
            kkk=0
            for e1 in pos_e:  
                e=int(e1)
                answerid=pos_e_answers[kkk]
                
                etext1=[]
                etext1=answers[int(answerid)][:100]
                etext=etext1
                
                for ii in range(len(etext)):
                    etext[ii]=vocab.index(etext[ii])+1
                    
                
                testlst=tq[0][0:2]
                testlst.append(user_id_map[str(e)])
                testlst=np.concatenate((testlst,[answerid],tq[0][2:]))        
                
                
                
                qid=testlst[0]
                answererid=testlst[2]
                qneighboirs=self.neighbors[qid].copy()
                #qneighboirs.remove(answererid)
                pos_eneigbors=self.neighbors[answererid].copy()
                
                
                
                
                
                #negative sample 
                e_neg=int(neg_e[kkk]) #get one of the negetive experts
                                
                etext1=[]    
                for aid in u_answers[int(e_neg)]:
                    #print(aid)
                    etext1.extend(answers[int(aid)][:100])
                    #etext1.extend(answers[int(aid)])

                #print("inter")
                #print(inter)
                etext_neg=etext1
                #etext=etext1
                if len(etext1)>max_d_len:                         
                        etext_neg=random.sample(etext1,max_d_len)
                        
                                
                for ii in range(len(etext_neg)):
                    etext_neg[ii]=vocab.index(etext_neg[ii])+1
                
                #if len(etext)<max_d_len:                
                    #for i in range(max_d_len-len(etext)):
                        #etext.append(0)
                
                testlst_neg=tq[0][0:2]
                testlst_neg.append(user_id_map[str(e_neg)])
                testlst_neg=np.concatenate((testlst_neg,[-1],tq[0][2:]))        
                
                answererid=testlst_neg[2]                
                neg_eneigbors=self.neighbors[answererid].copy()
                  
                
                val_data.append([testlst,testlst_neg])
                val_data_neighbors.append([[qneighboirs,pos_eneigbors],[qneighboirs,neg_eneigbors]])
                val_qatext.append([[qtext,etext],[qtext,etext_neg]])
                
                kkk+=1 
        return np.array(val_data),np.array(val_data_neighbors), np.array(val_qatext)   
    
    #adopted from knrm paper ref:https://github.com/AdeDZY/K-NRM
    @staticmethod
    def kernal_mus(n_kernels, use_exact):
        """
        get the mu for each guassian kernel. Mu is the middle of each bin
        :param n_kernels: number of kernels (including exact match). first one is exact match
        :return: l_mu, a list of mu.
        """
        if use_exact:
            l_mu = [1]
        else:
            l_mu = [2]
        if n_kernels == 1:
            return l_mu

        bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
        l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
        for i in range(1, n_kernels - 1):
            l_mu.append(l_mu[i] - bin_size)
        return l_mu

    #adopted from knrm paper copied from knrm paper ref:https://github.com/AdeDZY/K-NRM
    @staticmethod
    def kernel_sigmas(n_kernels, lamb, use_exact):
        """
        get sigmas for each guassian kernel.
        :param n_kernels: number of kernels (including exactmath.)
        :param lamb:
        :param use_exact:
        :return: l_sigma, a list of simga
        """
        bin_size = 2.0 / (n_kernels - 1)
        l_sigma = [0.00001]  # for exact match. small variance -> exact match
        if n_kernels == 1:
            return l_sigma

        l_sigma += [bin_size * lamb] * (n_kernels - 1)
        return l_sigma
    
    def q_a_rbf_words(self,inputs_q,inputs_d):  
        """text encoder \Psi"""
        # look up embeddings for each term. [nbatch, qlen, emb_dim]
        self.max_q_len=len(inputs_q[0])
        self.max_d_len=len(inputs_d[0])
        
        q_embed = tf.nn.embedding_lookup(self.wordembeddings, inputs_q, name='qemb')
        d_embed = tf.nn.embedding_lookup(self.wordembeddings, inputs_d, name='demb')
        batch_size=1
        
        # normalize and compute similarity matrix using l2 norm         
        norm_q = tf.sqrt(tf.reduce_sum(tf.square(q_embed), 2))
        #print(norm_q)
        norm_q=tf.reshape(norm_q,(len(norm_q),len(norm_q[0]),1))
        #print(norm_q)
        normalized_q_embed = q_embed / norm_q
        #print(normalized_q_embed)
        norm_d = tf.sqrt(tf.reduce_sum(tf.square(d_embed), 2))
        norm_d=tf.reshape(norm_d,(len(norm_d),len(norm_d[0]),1))
        normalized_d_embed = d_embed / norm_d
        #print(normalized_d_embed)
        tmp = tf.transpose(normalized_d_embed, perm=[0, 2, 1])
        #print(tmp)
        sim =tf.matmul(normalized_q_embed, tmp)
        #print(sim)        
        # compute gaussian kernel
        rs_sim = tf.reshape(sim, [batch_size, self.max_q_len, self.max_d_len, 1])
        #print(rs_sim)
        
        tmp = tf.exp(-tf.square(tf.subtract(rs_sim, self.mus)) / (tf.multiply(tf.square(self.sigmas), 2)))
        #print(tmp)
        
        feats = []  # store the soft-TF features from each field.
        # sum up gaussian scores
        kde = tf.reduce_sum(tmp, [2])
        kde = tf.math.log(tf.maximum(kde, 1e-10)) * 0.01  # 0.01 used to scale down the data.
        # [batch_size, qlen, n_bins]
        
        #print(kde)
        # aggregated query terms
        # q_weights = [1, 1, 0, 0...]. Works as a query word mask.
        # Support query-term weigting if set to continous values (e.g. IDF).
        
        #q_weights=np.where(np.array(inputs_q)>0,1,0)
        #q_weights=tf.dtypes.cast(q_weights, tf.float32)
        #q_weights = tf.reshape(q_weights, shape=[batch_size, self.max_q_len, 1])
        
        aggregated_kde = tf.reduce_sum(kde , [1])  # [batch, n_bins]   *q_weights
        #print( aggregated_kde)
        feats.append(aggregated_kde) # [[batch, nbins]]
        feats_tmp = tf.concat( feats,1)  # [batch, n_bins]
        #print ("batch feature shape:", feats_tmp.get_shape())
        
        # Reshape. (maybe not necessary...)
        feats_flat = tf.reshape(feats_tmp, [-1, self.n_bins])
        feats_flat2=tf.reshape(feats_flat, [1,self.n_bins])
        return(feats_flat2)  
    
    def q_a_rbf_nodes(self,inputs_q,inputs_d): 
        """sub-graph encoder \Phi"""        
        # look up embeddings for each term. [nbatch, qlen, emb_dim]
        self.max_q_len=len(inputs_q[0])
        self.max_d_len=len(inputs_d[0])
        
        q_embed = tf.nn.embedding_lookup(self.nodeembeddings, inputs_q, name='qemb')
        d_embed = tf.nn.embedding_lookup(self.nodeembeddings, inputs_d, name='demb')
        batch_size=1
        
        # normalize and compute similarity matrix using l2 norm         
        norm_q = tf.sqrt(tf.reduce_sum(tf.square(q_embed), 2))
        #print(norm_q)
        norm_q=tf.reshape(norm_q,(len(norm_q),len(norm_q[0]),1))
        #print(norm_q)
        normalized_q_embed = q_embed / norm_q
        #print(normalized_q_embed)
        norm_d = tf.sqrt(tf.reduce_sum(tf.square(d_embed), 2))
        norm_d=tf.reshape(norm_d,(len(norm_d),len(norm_d[0]),1))
        normalized_d_embed = d_embed / norm_d
        #print(normalized_d_embed)
        tmp = tf.transpose(normalized_d_embed, perm=[0, 2, 1])
        #print(tmp)
        sim =tf.matmul(normalized_q_embed, tmp)
        #print(sim)        
        # compute gaussian kernel
        rs_sim = tf.reshape(sim, [batch_size, self.max_q_len, self.max_d_len, 1])
        #print(rs_sim)
        
        tmp = tf.exp(-tf.square(tf.subtract(rs_sim, self.mus)) / (tf.multiply(tf.square(self.sigmas), 2)))
        #print(tmp)
        
        feats = []  # store the soft-TF features from each field.
        # sum up gaussian scores
        kde = tf.reduce_sum(tmp, [2])
        kde = tf.math.log(tf.maximum(kde, 1e-10)) * 0.01  # 0.01 used to scale down the data.
        # [batch_size, qlen, n_bins]
        
        #print(kde)
        # aggregated query terms
        # q_weights = [1, 1, 0, 0...]. Works as a query word mask.
        # Support query-term weigting if set to continous values (e.g. IDF).
        
        #q_weights=np.where(np.array(inputs_q)>0,1,0)
        #q_weights=tf.dtypes.cast(q_weights, tf.float32)
        #q_weights = tf.reshape(q_weights, shape=[batch_size, self.max_q_len, 1])
        
        aggregated_kde = tf.reduce_sum(kde , [1])  # [batch, n_bins]   *q_weights
        #print( aggregated_kde)
        feats.append(aggregated_kde) # [[batch, nbins]]
        feats_tmp = tf.concat( feats,1)  # [batch, n_bins]
        #print ("batch feature shape:", feats_tmp.get_shape())
        
        # Reshape. (maybe not necessary...)
        feats_flat = tf.reshape(feats_tmp, [-1, self.n_bins])
        feats_flat2=tf.reshape(feats_flat, [1,self.n_bins])
        return(feats_flat2)
        
               
    def model_test(self):
        embed=[]
        #print(self.inputs)
        for k in range(len(self.inputs)): 
            ind=self.inputs[k]
            qtext=[self.qatextinput[k][0]]
            atext=[self.qatextinput[k][1]]
            #print(qtext)
            #print(atext)
            q_a_rbf_words=self.q_a_rbf_words(qtext,atext)
            
            q_neighbors=[self.qaneighborinput[k][0]]
            a_neighbors=[self.qaneighborinput[k][1]]
            
            q_a_rbf_nodes=self.q_a_rbf_nodes(q_neighbors,a_neighbors)
                     
            embed1=tf.concat([q_a_rbf_nodes,q_a_rbf_words],1, name='concat')
            #embed1=tf.concat([qembed,askerembed,answererembed,tagsembed],1, name='concat')
            embed.append(embed1)
            
        embed=tf.reshape(embed,[len(self.inputs),self.regindim])    
        #return  tf.reshape(tf.matmul(embed,self.W4),[len(self.inputs)]) + self.b
        #print(embed)
        #print(len(embed))
        #print(len(embed[0]))
        w1out=tf.nn.tanh(tf.matmul(embed,self.W1))
        #print(w1out.shape)
        #w2out=tf.nn.tanh(tf.matmul(w1out,self.W2))
        #print(w2out.shape)
        #w3out=tf.nn.tanh(tf.matmul(w2out,self.W3))
        #print(w3out.shape)   
        return  tf.reshape(tf.matmul(w1out,self.W4),[len(self.inputs)]) + self.b
    
    def model(self,inputs,qatextinput,qaneighborinput):
        embed=[]
        #print(self.inputs)
        for k in range(len(inputs)): 
            ind=inputs[k]
            qtext=[qatextinput[k][0]]
            #print(qtext)
            atext=[qatextinput[k][1]]
            #print(atext)
            #print(qtext)
            #print(atext)
            #sys.exit(0)
            q_a_rbf_words=self.q_a_rbf_words(qtext,atext)
            
            q_neighbors=[qaneighborinput[k][0]]
            a_neighbors=[qaneighborinput[k][1]]
            
            q_a_rbf_nodes=self.q_a_rbf_nodes(q_neighbors,a_neighbors)
                     
            embed1=tf.concat([q_a_rbf_nodes,q_a_rbf_words],1, name='concat')
            #embed1=tf.concat([qembed,askerembed,answererembed,tagsembed],1, name='concat')
            embed.append(embed1)
        embed=tf.reshape(embed,[len(self.inputs),self.regindim])    
        #return  tf.reshape(tf.matmul(embed,self.W4),[len(self.inputs)]) + self.b
        #print(embed)
        #print(len(embed))
        #print(len(embed[0]))
        w1out=tf.nn.tanh(tf.matmul(embed,self.W1))
        #print(w1out.shape)
        #w2out=tf.nn.tanh(tf.matmul(w1out,self.W2))
        #print(w2out.shape)
        #w3out=tf.nn.tanh(tf.matmul(w2out,self.W3))
        #print(w3out.shape)   
        return  tf.reshape(tf.matmul(w1out,self.W4),[len(self.inputs)]) + self.b
        
    
    def loss(self):
        self.L= tf.reduce_mean(tf.math.maximum(0,1-self.model(self.inputs,self.qatextinput,self.qaneighborinput) 
                                         + self.model(self.inputs_neg,self.qatextinput_neg,self.qaneighborinput_neg)))
        return self.L  
        
    def train(self,modelname): 
        self.load_traindata(20,100)
        self.init_model()        
        print("train data loaded!!")     
        len_train_data=len(self.train_data)
        val_len=len(self.val_data)
        loss_=0
        epochs = range(50)
        self.batch_size=4
        global_step = tf.Variable(0, trainable=False)
        decayed_lr = tf.compat.v1.train.exponential_decay(0.0001,
                                        global_step, 700,
                                        0.95, staircase=True)
        opt = tf.keras.optimizers.Adam(learning_rate=decayed_lr,epsilon=6e-7)#(decayed_lr,epsilon=5e-6)
        logfile=open(self.dataset+"/results/log.txt","w")
        t_loss=[]
        v_loss=[]
        eps=[]
        
        for epoch in epochs:
            ind_new=[i for i in range(len_train_data)]
            np.random.shuffle(ind_new)
            self.train_data=self.train_data[ind_new,]
            self.train_label=self.train_label[ind_new,]           
            self.qatext=self.qatext[ind_new,]  
            self.train_data_neighbors=self.train_data_neighbors[ind_new,] 
            
            start=0
            end=0
            for i in range(math.ceil(len_train_data/self.batch_size)):
                if ((i+1)*self.batch_size)<len_train_data:                    
                    start=i*self.batch_size
                    end=(i+1)*self.batch_size
                else:                    
                    start=i*self.batch_size
                    end=len_train_data
                    
                self.inputs= [self.train_data[start:end][i][0] for i in range(len(self.train_data[start:end]))]                            
                self.qatextinput=[self.qatext[start:end][i][0] for i in range(len(self.qatext[start:end]))]
                self.qaneighborinput=[self.train_data_neighbors[start:end][i][0] for i in range(len( self.train_data_neighbors[start:end]))]
                
                self.inputs_neg= [self.train_data[start:end][i][1] for i in range(len(self.train_data[start:end]))]                            
                self.qatextinput_neg=[self.qatext[start:end][i][1] for i in range(len(self.qatext[start:end]))]
                self.qaneighborinput_neg=[self.train_data_neighbors[start:end][i][1] for i in range(len( self.train_data_neighbors[start:end]))]
                                             
                #print(i)
                #print(self.inputs)
                #print(self.outputs)
                #print(self.model())
                #print(self.qatextinput)
                
                opt.minimize(self.loss, var_list=[self.W1,self.W4,self.b,self.wordembeddings,self.nodeembeddings])#,self.W2,self.W3
                
                #q_embed = tf.nn.embedding_lookup(self.embeddings, self.qatextinput[0][0], name='qemb')
                #print(self.qatextinput[0][1])
                #print(self.outputs)
                #d_embed = tf.nn.embedding_lookup(self.embeddings, self.qatextinput[0][1], name='demb')
                #print(self.embeddings[0,:10])
                
                loss_+=self.L 
                
                global_step.assign_add(1)
                opt._decayed_lr(tf.float32)
                
                #print(self.Loss)
                #sys.exit(0)
                if (i+1)%50==0:                    
                    rep=(epoch*math.ceil(len_train_data/self.batch_size))+((i+1))
                    txt='Epoch %2d: i  %2d  out of  %4d     loss=%2.5f' %(epoch, i*self.batch_size, len_train_data, loss_/(rep))
                    logfile.write(txt+"\n")
                    print(txt)    
            #opt._decayed_lr(tf.float32)
            #print(self.W4)
            #validate the results
            print("\n************\nValidation started....\n")
            val_loss=0
            
            for ii in range(math.ceil(val_len/self.batch_size)):
                if ((ii+1)*self.batch_size)<val_len:
                    start=ii*self.batch_size
                    end=(ii+1)*self.batch_size
                else:
                    start=ii*self.batch_size
                    end=val_len
                
                self.inputs= [self.val_data[start:end][i][0] for i in range(len(self.val_data[start:end]))]                            
                self.qatextinput=[self.val_data_text[start:end][i][0] for i in range(len(self.val_data_text[start:end]))]
                self.qaneighborinput=[self.val_data_neighbors[start:end][i][0] for i in range(len( self.val_data_neighbors[start:end]))]
                
                self.inputs_neg= [self.val_data[start:end][i][1] for i in range(len(self.val_data[start:end]))]                            
                self.qatextinput_neg=[self.val_data_text[start:end][i][1] for i in range(len(self.val_data_text[start:end]))]
                self.qaneighborinput_neg=[self.val_data_neighbors[start:end][i][1] for i in range(len( self.val_data_neighbors[start:end]))]
                
                
                val_loss+=self.loss()
                #print(self.loss())
                #print(val_loss)
                if (ii+1)%50==0:                   
                    txt='Epoch %2d: ii  %2d  out of  %4d     validation loss=%2.5f' %(epoch, ii*self.batch_size, val_len, val_loss/(ii+1))
                    logfile.write(txt+"\n")
                    print(txt)
            txt='Epoch %2d: ii  %2d  out of  %4d     validation loss=%2.5f' %(epoch, ii*self.batch_size, val_len, val_loss/(ii+1))
            logfile.write(txt+"\n")
            print(txt)
            
            if epoch%1==0:
                pkl_filename =self.dataset+ "/results/pickle_QR_model.pkl"+str(epoch)+modelname
                with open(pkl_filename, 'wb') as file:
                    pickle.dump(self, file)
                print("model was saved")
            t_loss.append(loss_/(rep))
            v_loss.append(val_loss/math.ceil(val_len/self.batch_size))
            eps.append(epoch)
            plt.figure(figsize=(10,7))
            plt.plot(eps,t_loss,'r-o',label = "train loss")
            plt.plot(eps,v_loss,'b-*',label = "validation loss")
            plt.title("train and validation losses")
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(loc="upper right")
            plt.savefig(self.dataset+ "/results/loss.png")
            plt.show()
        print("train model done!!")
        logfile.close() 
        #print(self.W4)
        plt.figure(figsize=(10,7))
        plt.plot(eps,t_loss,'r-o',label = "train loss")
        plt.plot(eps,v_loss,'b-*',label = "validation loss")
        plt.title("train and validation losses")
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc="upper right")
        plt.savefig(self.dataset+ "/results/loss.png")
        plt.show()
    
    def test_model(dataset,modelname,path):        
        pkl_filename =dataset+ "/"+path+modelname
        # Load from file
        with open(pkl_filename, 'rb') as file:
            ob = pickle.load(file)
        print("model was loaded!!")
        #print(regr.get_params(deep=True))        
        #print(tf.reshape(ob.W4,(4,32)))
        #sys.exit(0)
        
        INPUT=dataset+"/test_data.txt"
        
        fin_test=open(INPUT)        
        test=fin_test.readline().strip()
        test_data=[]
        
        while test:
            data=test.split(";")
            lst=[]
            for d in data[0].split(" "):
                lst.append(int(d)) 
            
            alst=[]
            
            for d in data[1].split(" ")[0::3]:
                alst.append(int(d))
            
            anlst=[]
            for d in data[1].split(" ")[1::3]:
                anlst.append(int(d))
                
            test_data.append([lst,alst,anlst])
            
            test=fin_test.readline().strip()
        fin_test.close()       
        
        INPUT=dataset+"/CQG_proporties.txt"        
        pfile=open(INPUT)
        line=pfile.readline()
        N=int(line.split(" ")[2]) # number of nodes in the CQA network graph N=|Qestions|+|Askers|+|Answerers|+|tags|
        line=pfile.readline()
        qnum=int(line.split(" ")[2])
        
                
        user_id_map={}
        INPUT3=dataset+"/user_id_map.txt"
        fin=open(INPUT3, "r",encoding="utf8")
        line=fin.readline().strip()
        while line:            
            e=line.split(" ")
            uname=" ".join(e[1:])            
            uname=uname.strip()
            user_id_map[uname]=qnum+int(e[0])            
            line=fin.readline().strip()
            
        fin.close()
        
        
        
        answers={}
        qtitle={}
        qcontent={}
        vocab=[]
        
        INPUT=dataset+"/vocab.txt"
        fin=open( INPUT, "r")
        line=fin.readline()
        line=fin.readline().strip()
        while line:
            v = line.split(" ")        
            vocab.append(v[0])
            line=fin.readline().strip()
        
        INPUT=dataset+"/A_content_nsw.txt"
        with open( INPUT, "r") as fin:                
            for line in fin:
                d = line.strip().split(" ")        
                answers[int(d[0])]=d[1:]
        
        INPUT=dataset+"/Q_content_nsw.txt"
        with open( INPUT, "r") as fin:                
            for line in fin:
                d = line.strip().split(" ")        
                qcontent[int(d[0])]=d[1:]
        
        INPUT=dataset+"/Q_title_nsw.txt"
        with open( INPUT, "r") as fin:                
            for line in fin:
                d = line.strip().split(" ")        
                qtitle[int(d[0])]=d[1:] 
        
        Q_id_map_to_original={}
        INPUT2=dataset+"/Q_id_map.txt"
        ids=np.loadtxt(INPUT2, dtype=int)
        for e in ids:
            Q_id_map_to_original[int(e[0])]=int(e[1])
            
        max_q_len=ob.max_q_len
        ob.max_d_len=1*ob.max_d_len
        max_d_len=ob.max_d_len
        max_q_len=20
        max_d_len=100
        u_answers={}
        INPUT=dataset+"/user_answers.txt"
        with open( INPUT, "r") as fin:                
            for line in fin:
                d = line.strip().split(" ")        
                u_answers[int(d[0])]=d[1::2]
                
        
        batch_size=1        
        OUTPUT=dataset+"/"+path+"test_results_"+modelname+".txt"
        fout=open(OUTPUT,"w")
        #results=[]        
        iii=0
        for tq in test_data:
            print(iii)
            iii=iii+1
            print("test q:")
            print(tq)            
           
            ids=tq[1] 
            answerids=tq[2]
            
            print("experts:")      
            print(ids)
            inputs=[]
            inputtext=[]
            
            qtext=[]
            inputneighbors=[]
            
            qid=Q_id_map_to_original[int(tq[0][0])]
            qtext1=qtitle[qid].copy()
            qtext1.extend(qcontent[qid])
            qtext1=qtext1[:20]
            qtext=qtext1.copy()
            #print(qtext)
            for i in range(len(qtext)):
                qtext[i]=vocab.index(qtext[i])+1
            
            #if len(qtext)<max_q_len:                
            #        for i in range(max_q_len-len(qtext)):
                        #qtext.append(0)
            kkk=0
            for e in ids:              
                answerid=answerids[kkk]
                kkk+=1
                etext1=[]
                if answerid!=-1:
                    etext1=answers[int(answerid)][:100]
                    etext=etext1
                else:       
                    for aid in u_answers[int(e)]:
                        #print(aid)
                        etext1.extend(answers[int(aid)][:100])
                        #etext1.extend(answers[int(aid)])
                    
                    #print("inter")
                    #print(inter)
                    etext=etext1
                    #etext=etext1
                    if len(etext1)>max_d_len:                         
                            etext=random.sample(etext1,max_d_len)
                        
                
                #print(etext)
                
                for ii in range(len(etext)):
                    etext[ii]=vocab.index(etext[ii])+1
                
                #if len(etext)<max_d_len:                
                    #for i in range(max_d_len-len(etext)):
                        #etext.append(0)
                
                testlst=tq[0][0:2]
                testlst.append(user_id_map[str(e)])
                testlst=np.concatenate((testlst,tq[0][2:]))        
                inputs.append(testlst)
                inputtext.append([qtext,etext]) 
                
                qid1=testlst[0]
                #print(qid1)
                #print(ob.neighbors)
                answererid1=testlst[2]
                qneighboirs=ob.neighbors[qid1].copy()
                #qneighboirs.remove(answererid)
                eneigbors=ob.neighbors[answererid1].copy()
                
                inputneighbors.append([qneighboirs,eneigbors])
            ob.inputs=inputs
            ob.qatextinput=inputtext
            ob.qaneighborinput=inputneighbors
            #print(ob.inputs)
            #print(inputtext[0:2])
            s=ob.model_test().numpy() 
            print(s)
            res=""
            for i in range(len(ids)):
                res+=str(ids[i])+" "+ str(s[i])+";" 
            
            #res=" ".join([str(r) for r in sorted_ids[0:topk]])
            fout.write(res.strip()+"\n")
            fout.flush()
        fout.close()
        #OUTPUT=dataset+"/ColdEndFormat/EndCold_test_results.txt" 
        #np.savetxt(OUTPUT,np.array(results), fmt='%d')
        print("test_model done!!")           
    
    def test_model_allanswerers(dataset,modelname,path):        
        pkl_filename =dataset+ "/"+path+modelname
        # Load from file
        with open(pkl_filename, 'rb') as file:
            ob = pickle.load(file)
        print("model was loaded!!")
        #print(regr.get_params(deep=True))        
        #print(tf.reshape(ob.W4,(4,32)))
        #sys.exit(0)
        
        INPUT=dataset+"/"+"test_data.txt"
        
        fin_test=open(INPUT)        
        test=fin_test.readline().strip()
        test_data=[]
        
        while test:
            data=test.split(";")
            lst=[]
            for d in data[0].split(" "):
                lst.append(int(d)) 
            
            alst=[]
            
            for d in data[1].split(" ")[0::3]:
                alst.append(int(d))
            
            anlst=[]
            for d in data[1].split(" ")[1::3]:
                anlst.append(int(d))
                
            test_data.append([lst,alst,anlst])
            
            test=fin_test.readline().strip()
        fin_test.close()       
        
        INPUT=dataset+"/CQG_proporties.txt"        
        pfile=open(INPUT)
        line=pfile.readline()
        N=int(line.split(" ")[2]) # number of nodes in the CQA network graph N=|Qestions|+|Askers|+|Answerers|+|tags|
        line=pfile.readline()
        qnum=int(line.split(" ")[2])
        
                
        user_id_map={}
        INPUT3=dataset+"/user_id_map.txt"
        fin=open(INPUT3, "r",encoding="utf8")
        line=fin.readline().strip()
        while line:            
            e=line.split(" ")
            uname=" ".join(e[1:])            
            uname=uname.strip()
            user_id_map[uname]=qnum+int(e[0])            
            line=fin.readline().strip()
            
        fin.close()
        
        
        
        answers={}
        qtitle={}
        qcontent={}
        vocab=[]
        
        INPUT=dataset+"/vocab.txt"
        fin=open( INPUT, "r")
        line=fin.readline()
        line=fin.readline().strip()
        while line:
            v = line.split(" ")        
            vocab.append(v[0])
            line=fin.readline().strip()
        
        INPUT=dataset+"/A_content_nsw.txt"
        with open( INPUT, "r") as fin:                
            for line in fin:
                d = line.strip().split(" ")        
                answers[int(d[0])]=d[1:]
        
        INPUT=dataset+"/Q_content_nsw.txt"
        with open( INPUT, "r") as fin:                
            for line in fin:
                d = line.strip().split(" ")        
                qcontent[int(d[0])]=d[1:]
        
        INPUT=dataset+"/Q_title_nsw.txt"
        with open( INPUT, "r") as fin:                
            for line in fin:
                d = line.strip().split(" ")        
                qtitle[int(d[0])]=d[1:] 
        
        Q_id_map_to_original={}
        INPUT2=dataset+"/Q_id_map.txt"
        ids=np.loadtxt(INPUT2, dtype=int)
        for e in ids:
            Q_id_map_to_original[int(e[0])]=int(e[1])
            
        
        allanswererids=[]
        INPUT=dataset+"/user_tags.txt"
        fin=open(INPUT,"r")
        line=fin.readline()#skip file header
        line=fin.readline().strip()#read first line
        while line:
            allanswererids.append(int(line.split(" ")[0]))
            line=fin.readline().strip()
        fin.close()
        allanswererids=np.array(allanswererids)
        
        max_q_len=ob.max_q_len
        ob.max_d_len=1*ob.max_d_len
        max_d_len=ob.max_d_len
        max_q_len=20
        max_d_len=100
        u_answers={}
        INPUT=dataset+"/user_answers.txt"
        
        
        with open( INPUT, "r") as fin:                
            for line in fin:
                d = line.strip().split(" ")        
                u_answers[int(d[0])]=d[1::2]
                
                
        
        batch_size=1        
        OUTPUT=dataset+"/"+path+"test_results_all_"+modelname+".txt"
        fout=open(OUTPUT,"w")
        #results=[]        
        iii=0
        for tq in test_data:
            print(iii)
            iii=iii+1
            print("test q:")
            print(tq)
            
            alleids=list(np.setdiff1d(allanswererids,tq[1]))
            allaids=[-1]*len(alleids)
            
            ids=tq[1]
            ids.extend(alleids)
            answerids=tq[2]
            answerids.extend(allaids)
            
            print("experts:")      
            #print(ids)
            inputs=[]
            inputtext=[]
            inputneighbors=[]
            qtext=[]
            qid=Q_id_map_to_original[int(tq[0][0])]
            qtext1=qtitle[qid].copy()
            qtext1.extend(qcontent[qid])
            qtext1=qtext1[:20]
            qtext=qtext1.copy()
            #print(qtext)
            for i in range(len(qtext)):
                qtext[i]=vocab.index(qtext[i])+1
            
            #if len(qtext)<max_q_len:                
            #        for i in range(max_q_len-len(qtext)):
                        #qtext.append(0)
            kkk=0
            for e in ids:              
                answerid=answerids[kkk]
                kkk+=1
                etext1=[]
                if answerid!=-1:
                    etext1=answers[int(answerid)][:100]
                    etext=etext1
                else:       
                    for aid in u_answers[int(e)]:
                        #print(aid)
                        etext1.extend(answers[int(aid)][:100])
                        #etext1.extend(answers[int(aid)])
                    
                    #print("inter")
                    #print(inter)
                    etext=etext1
                    #etext=etext1
                    if len(etext1)>max_d_len:                         
                            etext=random.sample(etext1,max_d_len)
                        
                
                #print(etext)
                
                for ii in range(len(etext)):
                    etext[ii]=vocab.index(etext[ii])+1
                
                #if len(etext)<max_d_len:                
                    #for i in range(max_d_len-len(etext)):
                        #etext.append(0)
                
                testlst=tq[0][0:2]
                testlst.append(user_id_map[str(e)])
                testlst=np.concatenate((testlst,tq[0][2:]))        
                inputs.append(testlst)
                inputtext.append([qtext,etext]) 
                
                qid1=testlst[0]
                #print(qid1)
                #print(ob.neighbors)
                answererid1=testlst[2]
                qneighboirs=ob.neighbors[qid1].copy()
                #qneighboirs.remove(answererid)
                eneigbors=ob.neighbors[answererid1].copy()
                
                inputneighbors.append([qneighboirs,eneigbors])
            ob.inputs=inputs
            ob.qatextinput=inputtext
            ob.qaneighborinput=inputneighbors
            
                        
            #print(ob.inputs)
            #print(inputtext[0:2])
            s=ob.model_test().numpy() 
            print(s)
            res=""
            for i in range(len(ids)):
                res+=str(ids[i])+" "+ str(s[i])+";" 
            
            #res=" ".join([str(r) for r in sorted_ids[0:topk]])
            fout.write(res.strip()+"\n")
            fout.flush()
        fout.close()
        #OUTPUT=dataset+"/ColdEndFormat/EndCold_test_results.txt" 
        #np.savetxt(OUTPUT,np.array(results), fmt='%d')
        print("test_model done!!")    
    
dataset=["android","history","dba","physics"] 
data="../data/"+dataset[0]

#step 1
trian=False
if trian==True:
    ob=QRouting(data)
    ob.train("c") 
else: 
    option=["answerers+negativesamples","alluser"]
    #answerers+negativesamples: given test q, rank its true answerers plus some negative samples 
    #alluser: rank all experts given a test q
    op=option[1]
    if op==option[0]:
        for i in range(2,3):
             QRouting.test_model(data,"pickle_QR_model.pkl"+str(i)+"c","results/")
    elif op==option[1]:            
        QRouting.test_model_allanswerers(data,"pickle_QR_model.pkl0c","results/")
print("Done!")
        
