import torch
import torch.nn.functional as F
import copy
import time
import random
import operator
from attack.data_conversion import int2bin, bin2int
from quant.quantization import quan_Conv2d, quan_Linear, quantize
from sklearn.ensemble import IsolationForest
import numpy as np

class OGE(object):
    def __init__(self, criterion, k_top=10):
        self.criterion = criterion
        self.k_top = k_top
        self.loss_dict = {}
        self.bit_counter = 0  
        self.loss_max = 0
        self.n_generations = 50    
        self.population_size = 20  
        self.select_top = 10      
        self.offspring_count = 10  
        self.solution_size = 50    
        self.contamination = 0.01  
        self.loss = 0  

    def progressive_bit_search(self, model, data, target):
        """
        - Step 0: Evaluate current loss
        - Step 1: Identify vulnerable layers via gradient-based search
        - Step 2: Among the top vulnerable layers, gather outlier indices
        - Step 3: Use an Evolutionary search to pick the final bits to flip
        - Step 4: Perform the actual bit-flips on the model
        """
        model.eval()
        output = model(data)
        self.loss = self.criterion(output, target)
        for m in model.modules():
            if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
                if m.weight.grad is not None:
                    m.weight.grad.data.zero_()
        self.loss.backward()

        layer_losses = []
        idx = 0
        for name, module in model.named_modules():
            if isinstance(module, quan_Conv2d) or isinstance(module, quan_Linear):
                original_w = module.weight.data.detach().clone()
                w_grad = module.weight.grad.detach().flatten()
                k_actual = min(self.k_top, w_grad.numel())
                top_k_vals, top_k_inds = torch.topk(w_grad.abs(), k_actual)
                test_w = original_w.flatten().clone()
                for ii in top_k_inds.tolist():
                    test_w[ii] = -test_w[ii]
                test_w = test_w.view_as(module.weight)
                module.weight.data = test_w
                out_test = model(data)
                test_loss = self.criterion(out_test, target).item()
                module.weight.data.copy_(original_w)
                layer_losses.append((test_loss, name))
                idx += 1

        layer_losses.sort(reverse=True, key=lambda x: x[0])
        best_layer_name = layer_losses[0][1]
        self.loss_max = layer_losses[0][0]

        # ============= STEP 2: Gather outlier indices in best layer =============
        best_layer_module = None
        for nm, mod in model.named_modules():
            if nm == best_layer_name:
                best_layer_module = mod
                break
        if best_layer_module is None:
            return best_layer_name
        w_grad = best_layer_module.weight.grad.detach().flatten().cpu().numpy()
        iso = IsolationForest(contamination=self.contamination, n_estimators=100, random_state=42)
        w_grad = w_grad.reshape(-1,1)  
        iso_pred = iso.fit_predict(w_grad)
        outlier_inds = np.where(iso_pred == -1)[0]

        if len(outlier_inds) == 0:
            outlier_inds = np.random.choice(range(len(w_grad)), size=100, replace=False)

        # ============= STEP 3: Evolutionary search among outlier set =============
        self.loss_dict = {}
        original_w = best_layer_module.weight.data.detach().clone()
        device = original_w.device
        outlier_list = outlier_inds.tolist()
        def evaluate_solution(sol_inds):
            w_flat = original_w.flatten().clone()
            for idx_sol in sol_inds:
                w_flat[idx_sol] = -w_flat[idx_sol]
            best_layer_module.weight.data = w_flat.view_as(best_layer_module.weight)
            out = model(data)
            loss_val = self.criterion(out, target).item()

            return loss_val
        population = []
        for _ in range(self.population_size):
            candidate = random.sample(outlier_list, min(self.solution_size, len(outlier_list)))
            fit = evaluate_solution(candidate)
            population.append((candidate, fit))
        for gen in range(self.n_generations):
            population.sort(key=lambda x: x[1], reverse=True)  
            survivors = population[:self.select_top]
            new_population = survivors.copy()
            while len(new_population) < self.population_size:
                p1 = random.choice(survivors)
                p2 = random.choice(survivors)
                merged = list(set(p1[0] + p2[0]))
                if len(merged) > self.solution_size:
                    merged = random.sample(merged, self.solution_size)
                if random.random() < 0.2 and len(outlier_list) > 0:
                    mut_idx = random.choice(range(len(merged)))
                    merged[mut_idx] = random.choice(outlier_list)
                fit_c = evaluate_solution(merged)
                new_population.append((merged, fit_c))
            population = new_population

        population.sort(key=lambda x: x[1], reverse=True)
        best_solution, best_loss_val = population[0]

        # ============= STEP 4: Actually flip the bits in that best solution =============
        w_bin = int2bin(original_w.flatten(), best_layer_module.N_bits).short().clone()
        sign_mask = 1 << (best_layer_module.N_bits - 1)  
        for param_idx in best_solution:
            w_bin[param_idx] = w_bin[param_idx] ^ sign_mask
        param_flipped = bin2int(w_bin, best_layer_module.N_bits).view_as(original_w).float()
        best_layer_module.weight.data = param_flipped
        self.bit_counter += len(best_solution)
        return best_layer_name
