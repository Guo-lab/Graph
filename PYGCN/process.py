import numpy as np
import scipy.sparse as sp	
import torch





def load_data(path="./data/cora/", dataset="cora"):
    print('Loading {} dataset...'.format(dataset))
    
	#% content数据加载
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),dtype=np.dtype(str))
    
    # 获取特征向量，并将特征向量转为稀疏矩阵
    features = sp.csr_matrix( idx_features_labels[:, 1:-1], dtype=np.float32 )
    
    # 获取标签
    labels = encode_onehot( idx_features_labels[:, -1] )
    
    # 搭建图
    idx = np.array( idx_features_labels[:, 0], dtype=np.int32 )
    # 搭建字典 论文编号-->索引
    idx_map = {j: i for i, j in enumerate(idx)}
    
    #% cites数据加载 shape：5429,2
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),dtype=np.int32)
    
    # 边
    # 将编号映射为索引，因为编号是非连续的整数
    edges = np.array( list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    
    # 构建邻接矩阵
    adj = sp.coo_matrix(( np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1]) ),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    
	# 转换为对称邻接矩阵 
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
	
    # * 归一化特征矩阵和邻接矩阵
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
	
    # # 设置训练、验证和测试的数量
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)
	
    # 转为Tensor格式 [toarray返回ndarray todense返回矩阵]
    features = torch.FloatTensor(np.array(features.todense()))	
    labels = torch.LongTensor(np.where(labels)[1])				
    adj = sparse_mx_to_torch_sparse_tensor(adj)
	
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    #   adj 	2708,2708
    # features 	2708,1433
    #  labels	2708, 0~6	
    return adj, features, labels, idx_train, idx_val, idx_test












def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot



def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)















def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)