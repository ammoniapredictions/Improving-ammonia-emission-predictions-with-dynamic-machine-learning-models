import torch

def generatre_mini_batches(x_train, y_train, with_embeddings, batch_size, n_train, indices):
    
    mini_batches_x = []
    mini_batches_y = []
    
    if with_embeddings:

        x_cont_train_tmp = x_train[0]
        x_cat_train_tmp = x_train[1]
        
        for i in range (0, n_train, batch_size):
            x_cat_tmp = x_train[1] [indices[i : i + batch_size]]
            x_cat_tmp = torch.unbind (x_cat_tmp, dim = 1)
            mini_batches_x.append ([x_train[0][indices[i : i + batch_size]], x_cat_tmp])
    
    else:
    
        for i in range(0, n_train, batch_size):
            mini_batches_x.append (x_train[indices[i : i + batch_size]]) 
    

    for i in range(0, n_train, batch_size):
        mini_batches_y.append(y_train[indices[i : i + batch_size]])

    return [mini_batches_x, mini_batches_y]