#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[2]:


df_train = pd.read_csv('hw1_question1_train.csv', sep=',',header=None)
df_dev = pd.read_csv('hw1_question2_dev.csv', sep=',',header=None)
df_test = pd.read_csv('hw1_question2_test.csv', sep=',',header=None)

# add a column to answer the question is this sample malign? 
df_train[10] =df_train[9]>3
df_dev[10] =df_dev[9]>3
df_test[10] =df_test[9]>3


# # (a.i) Count examples in each class

# In[3]:



count = df_train[9].value_counts()
print("No. of Benign samples in training data: ", count[2])
print("No. of Malign samples in training data: ", count[4])


# # (a.ii) Data Exploration: Distribution of Features

# In[4]:


for i in range(0,9):
    sns.distplot(df_train[i], axlabel = "Feature " + str(i+1))
    plt.figure()


# # (a.iii) Scatterplots of Features 

# In[5]:


X = [0,6,1,5,4]
Y = [4,2,3,6,7]

for x_label,y_label in zip(X,Y):
        sns.scatterplot(x=x_label,y=y_label,hue=10,data=df_train,alpha=0.2)
        plt.xlabel("Feature " + str(x_label))
        plt.ylabel("Feature " + str(y_label))
        plt.figure()


# # (b.i) Implementation of KNN Classifier

# In[6]:


def knn_classifier(X_test,k,X_train,Y_train,distance_func):

    # apply to each row
    dist = np.apply_along_axis(distance_func, 1, X_train, X_test)
    # print(dist)

    # pick the k-closest
    idx = np.argpartition(dist, k)[:k]
    # print(idx)

    # will count the number of True examples in the idx selected by the selection
    pos = np.sum(Y_train[idx])
    # print(Y_train[idx])
    # print(pos)

    # check if more than half of the k examples selected are positive 
    if pos > (k/2):
      Y_test = True 
    else:
      Y_test = False

    return Y_test


# In[7]:


def euclidean(vec1,vec2):
  vec = (vec1 - vec2) ** 2
  ans = np.sum(vec)
  return np.sqrt(ans)


# In[8]:


# prepare the numpy arrays for faster predictions
X_train = df_train.iloc[:,:9].values
Y_train =  df_train.iloc[:,10].values
X_dev = df_dev.iloc[:,:9].values
Y_dev =  df_dev.iloc[:,10].values


# In[9]:


results = []
for k in range(1,20,2):
  # apply to each row in our X_dev
  predictions = np.apply_along_axis(knn_classifier, 1, X_dev,k,X_train,Y_train,euclidean)

  acc = np.sum(predictions==Y_dev)/np.shape(Y_dev)

  pos_idx = np.where(Y_dev==True)
  neg_idx = np.where(Y_dev==False)
  bcc = 0.5 *  np.sum(predictions[pos_idx]==Y_dev[pos_idx])/np.shape(Y_dev[pos_idx])
  bcc += 0.5 * np.sum(predictions[neg_idx]==Y_dev[neg_idx])/np.shape(Y_dev[neg_idx])
  results.append([k,acc[0],bcc[0]])


# # (b.ii) Metrics with different K values

# In[10]:


for result in results:
  print("K=",result[0]," Acc=",result[1],"Bcc=",result[2])


# In[11]:


# transform to a numpy array
results_np = np.asarray(results)
sns.set("paper","whitegrid")

plt.plot(results_np[:,0],results_np[:,1],label="ACC")
plt.plot(results_np[:,0],results_np[:,2],label="BCC")

plt.xticks(results_np[:,0])
plt.legend()

plt.title('Development set accuracy')
plt.xlabel('K')
plt.ylabel('Accuracy')

plt.show()


# In[12]:


acc_sorted = sorted(results,key=lambda x:x[1],reverse=True)
bcc_sorted = sorted(results,key=lambda x:x[2],reverse=True)

K1 = acc_sorted[0][0]
K2 = bcc_sorted[0][0]

print("Best Accuracy K1 = ",K1)
print("Best Balanced Accuracy K2 = ",K2)


# # (b.iii) Metrics on Test set using K1 and K2

# In[13]:


# Calculate the accuracies for test set with this K1 and K2
X_test = df_test.iloc[:,:9].values
Y_test =  df_test.iloc[:,10].values

Best_Ks = [K1,K2]
final_results = []

for k in Best_Ks:
  predictions = np.apply_along_axis(knn_classifier, 1, X_test,k,X_train,Y_train,euclidean)

  final_acc = np.sum(predictions==Y_test)/np.shape(Y_test)

  pos_idx = np.where(Y_test==True)
  neg_idx = np.where(Y_test==False)
  final_bcc = 0.5 *  np.sum(predictions[pos_idx]==Y_test[pos_idx])/np.shape(Y_test[pos_idx])
  final_bcc += 0.5 * np.sum(predictions[neg_idx]==Y_test[neg_idx])/np.shape(Y_test[neg_idx])
  final_results.append([final_acc[0],final_bcc[0]])


# In[14]:


# report the results
final_results = np.asarray(final_results)
labels = ['ACC', 'BCC']

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, final_results[0,:], width, label='K1')
rects2 = ax.bar(x + width/2, final_results[1,:], width, label='K2')

# Add some text for labels, title and custom x-axis tick labels, etc.
sns.set("paper","whitegrid")
ax.set_ylabel('Scores')
ax.set_title('Accuracy Metrics Comparision for K1 and K2 on Test Set')
ax.set_ylim([0,1.2])
ax.set_yticks(np.arange(0.0, 1.1, 0.1))
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc=1)

plt.figure()


# In[15]:


compare_dist = []
compare_dist.append(["Euclidean",final_results[0,:]])
print(compare_dist)


# # (b.iv) Bonus: Comparison of different distance computations

# In[16]:


## Storing the Euclidean distance resutls:
compare_dist = []
compare_dist.append(["Euclidean",final_results[0,:]])


# In[17]:


def manhattan(vec1,vec2):
  vec = (vec1 - vec2)
  ans = np.sum(np.absolute(vec))
  return ans


# In[18]:


# Get the Best Manhattan score
man_results = []
for k in range(1,20,2):
  # apply to each row in our X_dev
  predictions = np.apply_along_axis(knn_classifier, 1, X_dev,k,X_train,Y_train,manhattan)

  acc = np.sum(predictions==Y_dev)/np.shape(Y_dev)

  pos_idx = np.where(Y_dev==True)
  neg_idx = np.where(Y_dev==False)
  bcc = 0.5 *  np.sum(predictions[pos_idx]==Y_dev[pos_idx])/np.shape(Y_dev[pos_idx])
  bcc += 0.5 * np.sum(predictions[neg_idx]==Y_dev[neg_idx])/np.shape(Y_dev[neg_idx])
  man_results.append([k,acc[0],bcc[0]])


# In[19]:


# how does our accuracy change with k?
man_results_np = np.asarray(man_results)
sns.set("paper","whitegrid")

plt.plot(man_results_np[:,0],man_results_np[:,1],label="ACC")
plt.plot(man_results_np[:,0],man_results_np[:,2],label="BCC")

plt.xticks(man_results_np[:,0])
plt.legend()

plt.title('Development set accuracy- Manhattan Distance')
plt.xlabel('K')
plt.ylabel('Accuracy')

plt.show()


# In[20]:


man_acc_sorted = sorted(man_results,key=lambda x:x[1],reverse=True)
man_K1 = man_acc_sorted[0][0]

predictions = np.apply_along_axis(knn_classifier, 1, X_test,man_K1,X_train,Y_train,manhattan)

man_acc = np.sum(predictions==Y_test)/np.shape(Y_test)

pos_idx = np.where(Y_test==True)
neg_idx = np.where(Y_test==False)
man_bcc = 0.5 *  np.sum(predictions[pos_idx]==Y_test[pos_idx])/np.shape(Y_test[pos_idx])
man_bcc += 0.5 * np.sum(predictions[neg_idx]==Y_test[neg_idx])/np.shape(Y_test[neg_idx])
compare_dist.append(["Manhattan",np.asarray([man_acc[0],man_bcc[0]])])


# In[21]:


## Cosine similarity as a distance measure

def cos_sim(vec1,vec2):
  ans = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
  return ans


# In[22]:


# Get the Best Cosine score
cos_results = []
for k in range(1,20,2):
  # apply to each row in our X_dev
  predictions = np.apply_along_axis(knn_classifier, 1, X_dev,k,X_train,Y_train,cos_sim)

  acc = np.sum(predictions==Y_dev)/np.shape(Y_dev)

  pos_idx = np.where(Y_dev==True)
  neg_idx = np.where(Y_dev==False)
  bcc = 0.5 *  np.sum(predictions[pos_idx]==Y_dev[pos_idx])/np.shape(Y_dev[pos_idx])
  bcc += 0.5 * np.sum(predictions[neg_idx]==Y_dev[neg_idx])/np.shape(Y_dev[neg_idx])
  cos_results.append([k,acc[0],bcc[0]])


# In[23]:


# how does our accuracy change with k?
cos_results_np = np.asarray(cos_results)
sns.set("paper","whitegrid")

plt.plot(cos_results_np[:,0],cos_results_np[:,1],label="ACC")
plt.plot(cos_results_np[:,0],cos_results_np[:,2],label="BCC")

plt.xticks(cos_results_np[:,0])
plt.legend()

plt.title('Development set accuracy- Cosine Similarity Distance')
plt.xlabel('K')
plt.ylabel('Accuracy')

plt.show()


# In[24]:


cos_acc_sorted = sorted(cos_results,key=lambda x:x[1],reverse=True)
cos_K1 = cos_acc_sorted[0][0]

predictions = np.apply_along_axis(knn_classifier, 1, X_test,cos_K1,X_train,Y_train,cos_sim)

cos_acc = np.sum(predictions==Y_test)/np.shape(Y_test)

pos_idx = np.where(Y_test==True)
neg_idx = np.where(Y_test==False)
cos_bcc = 0.5 *  np.sum(predictions[pos_idx]==Y_test[pos_idx])/np.shape(Y_test[pos_idx])
cos_bcc += 0.5 * np.sum(predictions[neg_idx]==Y_test[neg_idx])/np.shape(Y_test[neg_idx])
compare_dist.append(["Cosine",np.asarray([cos_acc[0],cos_bcc[0]])])


# In[25]:


## chi-square distance

def chi_sqr(vec1,vec2):
  vec = (vec1 - vec2) ** 2
  vec = vec/np.absolute(vec1+vec2+1e-4)
  ans = np.sum(vec)
  return np.sqrt(ans)


# In[26]:


# Get the Best Cosine score
chi_results = []
for k in range(1,20,2):
  # apply to each row in our X_dev
  predictions = np.apply_along_axis(knn_classifier, 1, X_dev,k,X_train,Y_train,chi_sqr)

  acc = np.sum(predictions==Y_dev)/np.shape(Y_dev)

  pos_idx = np.where(Y_dev==True)
  neg_idx = np.where(Y_dev==False)
  bcc = 0.5 *  np.sum(predictions[pos_idx]==Y_dev[pos_idx])/np.shape(Y_dev[pos_idx])
  bcc += 0.5 * np.sum(predictions[neg_idx]==Y_dev[neg_idx])/np.shape(Y_dev[neg_idx])
  chi_results.append([k,acc[0],bcc[0]])


# In[27]:


# how does our accuracy change with k?
chi_results_np = np.asarray(chi_results)
sns.set("paper","whitegrid")

plt.plot(chi_results_np[:,0],chi_results_np[:,1],label="ACC")
plt.plot(chi_results_np[:,0],chi_results_np[:,2],label="BCC")

plt.xticks(chi_results_np[:,0])
plt.legend()

plt.title('Development set accuracy- Chi-Square Distance')
plt.xlabel('K')
plt.ylabel('Accuracy')

plt.show()


# In[28]:


chi_acc_sorted = sorted(chi_results,key=lambda x:x[1],reverse=True)
chi_K1 = chi_acc_sorted[0][0]

predictions = np.apply_along_axis(knn_classifier, 1, X_test,chi_K1,X_train,Y_train,chi_sqr)

chi_acc = np.sum(predictions==Y_test)/np.shape(Y_test)

pos_idx = np.where(Y_test==True)
neg_idx = np.where(Y_test==False)
chi_bcc = 0.5 *  np.sum(predictions[pos_idx]==Y_test[pos_idx])/np.shape(Y_test[pos_idx])
chi_bcc += 0.5 * np.sum(predictions[neg_idx]==Y_test[neg_idx])/np.shape(Y_test[neg_idx])
compare_dist.append(["Chi-Square",np.asarray([chi_acc[0],chi_bcc[0]])])


# In[29]:


print(compare_dist)


# In[30]:


# report the results
labels = ['ACC', 'BCC']

x = np.arange(len(labels))  # the label locations
width = 0.20  # the width of the bars

fig, ax = plt.subplots()
i = 0

for dist_res in compare_dist:
  rects1 = ax.bar(x + width*i, dist_res[1], width, label=dist_res[0])
  i = i+1

# rects2 = ax.bar(x + width/2, final_results[1,:], width, label='K2')

# Add some text for labels, title and custom x-axis tick labels, etc.
sns.set("paper","whitegrid")
ax.set_ylabel('Scores')
ax.set_title('Accuracy Metrics Comparision for K1 and K2 on Test Set')
ax.set_ylim([0,1.4])
ax.set_yticks(np.arange(0.0, 1.1, 0.1))
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc=1)

plt.figure()

