import torch 
import string
import re

def get_target_save_modules(model):
    """
    Gets all the column attention linear layers (k_proj, v_proj, q_proj and out_proj) for LoRA fine-tuning.
    Also, get modules to save, i.e. the modules that are will be trained as well but not with LoRA (in our case FCN layers on top).
    
    Args: 
        model (nn.Module): Model from which we want to extract linear layers.
    
    Returns:
        store_target_modules (list): List containing all the target modules.
    """
    
    store_target_modules = []
    store_modules_to_save = []

    for n, m in model.named_modules():
        
        # If there is a dot in the name of the module
        if '.' in n:
            
            split_string = n.split('.')
            
            # Must satisfy this contraint
            if ('column_self_attention' in split_string) and (('k_proj' in split_string) or ('v_proj' in split_string) or ('q_proj' in split_string) or ('out_proj' in split_string)):
                store_target_modules.append(n)
                
            if get_modules_to_save(split_string):
                store_modules_to_save.append(n)
            
    return store_target_modules, store_modules_to_save

def get_modules_to_save(name):
    """
    Check if elements in the list match patterns. 
    
    Args:
        name (list of str): List of elements of string when split on '.'
        
    Return: 
        satisfy (bool): Whether it satisfies the pattern or not. 
    """
    # Define bool to indicate whether there is pattern in matching 
    satisfy = False
    
    # Regular expression pattern to match
    pattern = r"finetune_linear_\d+"
    
    if any(re.match(pattern, s) for s in name):
        satisfy = True

    return satisfy    