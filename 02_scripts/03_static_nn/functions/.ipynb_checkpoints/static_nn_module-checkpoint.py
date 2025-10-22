
import torch
import torch.nn as nn



class RegressionNN (nn.Module):
    
    
    def __init__(self, layer_sizes, with_embeddings = False, cat_dims = None, embeddings_dims = None):
        
        super (RegressionNN, self).__init__()

        self.with_embeddings = with_embeddings

        if (self.with_embeddings):
       
            self.embeddings = nn.ModuleList ([
                nn.Embedding (num_embeddings = cat_dim, embedding_dim = embed_dim)
                for cat_dim, embed_dim in zip (cat_dims, embeddings_dims)
            ])
            
            layer_sizes[0] = layer_sizes[0] - len (cat_dims) + sum (embeddings_dims)
        
        layers = []
        
        for i in range (len (layer_sizes) - 1):
            
            layers.append (nn.Linear (layer_sizes[i], layer_sizes[i + 1]))
            
            if i < len (layer_sizes) - 2:  # No ReLU after the last linear layer
                
                layers.append (nn.ReLU())

        self.model = nn.Sequential (*layers)
 

    def forward (self, x):

        if (self.with_embeddings):

            x_continuous = x[0]
            x_categoricals = x[1]
            
            x_embeds = [embed (x_cat) for embed, x_cat in zip (self.embeddings, x_categoricals)]
            
            x = torch.cat ([x_continuous] + x_embeds, dim = -1)

        
        return self.model (x)
