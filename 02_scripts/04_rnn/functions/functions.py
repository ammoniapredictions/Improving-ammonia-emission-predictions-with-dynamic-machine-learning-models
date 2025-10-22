import torch


def generate_tensors_response(df, pmid, response, device):
    
    data_filtered = df[df['pmid'] == pmid]
    
    if (response == "e.cum and delta_e.cum"):
        response_tensor = torch.tensor(data_filtered[['e.cum', 'delta_e.cum']].values, dtype=torch.float32)
    
    if (response == "e.cum"):
        response_tensor = torch.tensor(data_filtered[['e.cum']].values, dtype=torch.float32)
    
    if (response == "delta_e.cum"):
        response_tensor = torch.tensor(data_filtered[['delta_e.cum']].values, dtype=torch.float32)
    
    return response_tensor.to(device)



def generate_tensors_predictors(df, pmid, with_embeddings, device):
    
    data_filtered = df[df['pmid'] == pmid]

    if (with_embeddings):
        x_cont = data_filtered[['ct', 'dt', 'air.temp', 'wind.2m', 'rain.rate', 'tan.app', 'app.rate', 'man.dm', 'man.ph', 't.incorp']]
    
        x_cont_tensor = torch.tensor(x_cont.values, dtype=torch.float32).view(len(x_cont), len(x_cont.columns))
        x_cont_tensor = x_cont_tensor.to(device)
        
        x_cat = data_filtered[['app.mthd', 'incorp', 'man.source']]
        
        x_cat_tensor = torch.tensor(x_cat.values, dtype=torch.long).view(len(x_cat), len(x_cat.columns))
        x_cat_tensor = x_cat_tensor.to(device)
        x_cat_tensor = torch.unbind (x_cat_tensor, dim = 1)

        output = [x_cont_tensor, x_cat_tensor]

    else: 
                    
        predictors = data_filtered[['ct', 'dt', 'air.temp', 'wind.2m', 'rain.rate', 'tan.app', 'app.rate', 'man.dm', 'man.ph', 't.incorp', 'app.mthd_bc', 'app.mthd_bsth',
                                   'app.mthd_ts', 'app.mthd_os', 'man.source', 'incorp_none', 'incorp_shallow']]
        
        predictors_tensor = torch.tensor(predictors.values, dtype=torch.float32).view(len(predictors), len(predictors.columns))
        output = predictors_tensor.to(device)
    
    return output


def compute_mae_eval(response, x_eval, target_eval, model, n_observations_evaluation_subset, device):
    
    all_predictions = torch.empty(0).to(device)
    
    with torch.no_grad():

        for x in x_eval:
                        
            y = model (x)
        
            all_predictions = torch.cat ((all_predictions, y.squeeze()), 0)

        mae = torch.sum (torch.abs (all_predictions - target_eval)) / n_observations_evaluation_subset

        if (response == "e.cum and delta_e.cum"):
            
            mae_ecum = torch.sum (torch.abs (all_predictions[:,0] - target_eval[:,0])) / n_observations_evaluation_subset
            mae_delta_ecum = torch.sum (torch.abs (all_predictions[:,1] - target_eval[:,1])) / n_observations_evaluation_subset    
    
            return [mae.item(), mae_ecum.item(), mae_delta_ecum.item()]

        else: 
            
            return [mae.item()] 




# for interpolated data (dt is removed of the list of the predictors since it is constant)
def generate_tensors_predictors_2(df, pmid, with_embeddings, device):
    
    data_filtered = df[df['pmid'] == pmid]

    if (with_embeddings):
        x_cont = data_filtered[['ct', 'air.temp', 'wind.2m', 'rain.rate', 'tan.app', 'app.rate', 'man.dm', 'man.ph', 't.incorp']]
    
        x_cont_tensor = torch.tensor(x_cont.values, dtype=torch.float32).view(len(x_cont), len(x_cont.columns))
        x_cont_tensor = x_cont_tensor.to(device)
        
        x_cat = data_filtered[['app.mthd', 'incorp', 'man.source']]
        
        x_cat_tensor = torch.tensor(x_cat.values, dtype=torch.long).view(len(x_cat), len(x_cat.columns))
        x_cat_tensor = x_cat_tensor.to(device)
        x_cat_tensor = torch.unbind (x_cat_tensor, dim = 1)

        output = [x_cont_tensor, x_cat_tensor]

    else: 
                    
        predictors = data_filtered[['ct', 'air.temp', 'wind.2m', 'rain.rate', 'tan.app', 'app.rate', 'man.dm', 'man.ph', 't.incorp', 'app.mthd_bc', 'app.mthd_bsth',
                                   'app.mthd_ts', 'app.mthd_os', 'man.source', 'incorp_none', 'incorp_shallow']]
        
        predictors_tensor = torch.tensor(predictors.values, dtype=torch.float32).view(len(predictors), len(predictors.columns))
        output = predictors_tensor.to(device)
    
    return output



