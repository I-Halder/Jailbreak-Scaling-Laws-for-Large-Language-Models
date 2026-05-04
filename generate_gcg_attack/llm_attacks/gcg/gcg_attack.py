import gc

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from llm_attacks import AttackPrompt, MultiPromptAttack, MultiPromptAttackCoherence, PromptManager
from llm_attacks import get_embedding_matrix, get_embeddings


def token_gradients(model, input_ids, input_slice, target_slice, loss_slice):

    """
    Computes gradients of the loss with respect to the coordinates.
    
    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    input_slice : slice
        The slice of the input sequence for which gradients need to be computed.
    target_slice : slice
        The slice of the input sequence to be used as targets.
    loss_slice : slice
        The slice of the logits to be used for computing the loss.

    Returns
    -------
    torch.Tensor
        The gradients of each token in the input_slice with respect to the loss.
    """

    embed_weights = get_embedding_matrix(model)
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1, 
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)
    
    # now stitch it together with the rest of the embeddings
    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:,:input_slice.start,:], 
            input_embeds, 
            embeds[:,input_slice.stop:,:]
        ], 
        dim=1)
    
    logits = model(inputs_embeds=full_embeds).logits
    targets = input_ids[target_slice]
    loss = nn.CrossEntropyLoss()(logits[0,loss_slice,:], targets)
    
    loss.backward()
    
    return one_hot.grad.clone()

class GCGAttackPrompt(AttackPrompt):

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
    
    def grad(self, model):
        return token_gradients(
            model, 
            self.input_ids.to(model.device), 
            self._control_slice, 
            self._target_slice, 
            self._loss_slice
        )

class GCGPromptManager(PromptManager):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def sample_control(self, grad, batch_size, topk=256, temp=1, allow_non_ascii=True):

        if not allow_non_ascii:
            grad[:, self._nonascii_toks.to(grad.device)] = np.infty
        top_indices = (-grad).topk(topk, dim=1).indices
        control_toks = self.control_toks.to(grad.device)
        original_control_toks = control_toks.repeat(batch_size, 1)
        new_token_pos = torch.arange(
            0, 
            len(control_toks), 
            len(control_toks) / batch_size,
            device=grad.device
        ).type(torch.int64)
        new_token_val = torch.gather(
            top_indices[new_token_pos], 1, 
            torch.randint(0, topk, (batch_size, 1),
            device=grad.device)
        )
        new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)
        return new_control_toks


class GCGMultiPromptAttack(MultiPromptAttack):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def step(self, 
             batch_size=1024, 
             topk=256, 
             temp=1, 
             allow_non_ascii=True, 
             target_weight=1, 
             control_weight=0.1, 
             verbose=False, 
             opt_only=False,
             filter_cand=True):

        
        # GCG currently does not support optimization_only mode, 
        # so opt_only does not change the inner loop.
        opt_only = False

        main_device = self.models[0].device
        control_cands = []

        for j, worker in enumerate(self.workers):
            worker(self.prompts[j], "grad", worker.model)

        # Aggregate gradients
        grad = None
        for j, worker in enumerate(self.workers):
            new_grad = worker.results.get().to(main_device)
            new_grad = new_grad / new_grad.norm(dim=-1, keepdim=True)
            if grad is None:
                grad = torch.zeros_like(new_grad)
            if grad.shape != new_grad.shape:
                with torch.no_grad():
                    control_cand = self.prompts[j-1].sample_control(grad, batch_size, topk, temp, allow_non_ascii)
                    control_cands.append(self.get_filtered_cands(j-1, control_cand, filter_cand=filter_cand, curr_control=self.control_str))
                grad = new_grad
            else:
                grad += new_grad

        with torch.no_grad():
            control_cand = self.prompts[j].sample_control(grad, batch_size, topk, temp, allow_non_ascii)
            control_cands.append(self.get_filtered_cands(j, control_cand, filter_cand=filter_cand, curr_control=self.control_str))
        del grad, control_cand ; gc.collect()
        
        # Search
        loss = torch.zeros(len(control_cands) * batch_size).to(main_device)
        with torch.no_grad():
            for j, cand in enumerate(control_cands):
                # Looping through the prompts at this level is less elegant, but
                # we can manage VRAM better this way
                progress = tqdm(range(len(self.prompts[0])), total=len(self.prompts[0])) if verbose else enumerate(self.prompts[0])
                for i in progress:
                    for k, worker in enumerate(self.workers):
                        worker(self.prompts[k][i], "logits", worker.model, cand, return_ids=True)
                    logits, ids = zip(*[worker.results.get() for worker in self.workers])
                    loss[j*batch_size:(j+1)*batch_size] += sum([
                        target_weight*self.prompts[k][i].target_loss(logit, id).mean(dim=-1).to(main_device) 
                        for k, (logit, id) in enumerate(zip(logits, ids))
                    ])
                    if control_weight != 0:
                        loss[j*batch_size:(j+1)*batch_size] += sum([
                            control_weight*self.prompts[k][i].control_loss(logit, id).mean(dim=-1).to(main_device)
                            for k, (logit, id) in enumerate(zip(logits, ids))
                        ])
                    del logits, ids ; gc.collect()
                    
                    if verbose:
                        progress.set_description(f"loss={loss[j*batch_size:(j+1)*batch_size].min().item()/(i+1):.4f}")

            min_idx = loss.argmin()
            model_idx = min_idx // batch_size
            batch_idx = min_idx % batch_size
            next_control, cand_loss = control_cands[model_idx][batch_idx], loss[min_idx]
        
        del control_cands, loss ; gc.collect()

        print('Current length:', len(self.workers[0].tokenizer(next_control).input_ids[1:]))
        print(next_control)

        return next_control, cand_loss.item() / len(self.prompts[0]) / len(self.workers)
    
    
class GCGMultiPromptAttackCoherence(MultiPromptAttackCoherence):
    """
    GCG-specific implementation of multi-prompt attack with coherence regularization.
    Inherits from MultiPromptAttackCoherence base class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, 
             batch_size=1024, 
             topk=256, 
             temp=1, 
             allow_non_ascii=True, 
             target_weight=1, 
             control_weight=0.1, 
             perplexity_weight=0.05,  # New parameter for perplexity regularization
             verbose=False, 
             opt_only=False,
             filter_cand=True):

        
        # GCG currently does not support optimization_only mode, 
        # so opt_only does not change the inner loop.
        opt_only = False

        main_device = self.models[0].device
        control_cands = []

        for j, worker in enumerate(self.workers):
            worker(self.prompts[j], "grad", worker.model)

        # Aggregate gradients
        grad = None
        for j, worker in enumerate(self.workers):
            new_grad = worker.results.get().to(main_device)
            new_grad = new_grad / new_grad.norm(dim=-1, keepdim=True)
            if grad is None:
                grad = torch.zeros_like(new_grad)
            if grad.shape != new_grad.shape:
                with torch.no_grad():
                    control_cand = self.prompts[j-1].sample_control(grad, batch_size, topk, temp, allow_non_ascii)
                    control_cands.append(self.get_filtered_cands(j-1, control_cand, filter_cand=filter_cand, curr_control=self.control_str))
                grad = new_grad
            else:
                grad += new_grad

        with torch.no_grad():
            control_cand = self.prompts[j].sample_control(grad, batch_size, topk, temp, allow_non_ascii)
            control_cands.append(self.get_filtered_cands(j, control_cand, filter_cand=filter_cand, curr_control=self.control_str))
        del grad, control_cand ; gc.collect()
        
        # Search
        loss = torch.zeros(len(control_cands) * batch_size).to(main_device)
        with torch.no_grad():
            for j, cand in enumerate(control_cands):
                # Looping through the prompts at this level is less elegant, but
                # we can manage VRAM better this way
                progress = tqdm(range(len(self.prompts[0])), total=len(self.prompts[0])) if verbose else enumerate(self.prompts[0])
                for i in progress:
                    for k, worker in enumerate(self.workers):
                        worker(self.prompts[k][i], "logits", worker.model, cand, return_ids=True)
                    logits, ids = zip(*[worker.results.get() for worker in self.workers])
                    
                    # Original loss components
                    loss[j*batch_size:(j+1)*batch_size] += sum([
                        target_weight*self.prompts[k][i].target_loss(logit, id).mean(dim=-1).to(main_device) 
                        for k, (logit, id) in enumerate(zip(logits, ids))
                    ])
                    if control_weight != 0:
                        loss[j*batch_size:(j+1)*batch_size] += sum([
                            control_weight*self.prompts[k][i].control_loss(logit, id).mean(dim=-1).to(main_device)
                            for k, (logit, id) in enumerate(zip(logits, ids))
                        ])
                    
                    # NEW: Perplexity-based coherence loss
                    if perplexity_weight != 0:
                        perplexity_losses = []
                        for k, (logit, id) in enumerate(zip(logits, ids)):
                            # Calculate perplexity for the control/adversarial suffix
                            perplexity_loss = self.calculate_perplexity_loss(
                                self.prompts[k][i], logit, id, cand
                            )
                            perplexity_losses.append(perplexity_loss)
                        
                        loss[j*batch_size:(j+1)*batch_size] += sum([
                            perplexity_weight * ppl_loss.mean(dim=-1).to(main_device)
                            for ppl_loss in perplexity_losses
                        ])
                        
                        del perplexity_losses  # Clean up perplexity losses
                    
                    del logits, ids ; gc.collect()
                    
                    if verbose:
                        progress.set_description(f"loss={loss[j*batch_size:(j+1)*batch_size].min().item()/(i+1):.4f}")

            min_idx = loss.argmin()
            model_idx = min_idx // batch_size
            batch_idx = min_idx % batch_size
            next_control, cand_loss = control_cands[model_idx][batch_idx], loss[min_idx]
        
        del control_cands, loss ; gc.collect()

        print('Current length:', len(self.workers[0].tokenizer(next_control).input_ids[1:]))
        print(next_control)

        return next_control, cand_loss.item() / len(self.prompts[0]) / len(self.workers)

    # def calculate_perplexity_loss(self, prompt, logits, ids, cand):
    #     """
    #     Calculate perplexity loss for the control/adversarial suffix to encourage coherence.
    #     Lower perplexity indicates better coherence.
    #     """
    #     import torch.nn.functional as F
        
    #     # Get the tokenizer from the worker (assuming first worker's tokenizer)
    #     tokenizer = self.workers[0].tokenizer
        
    #     # Tokenize the candidate control string to get its length
    #     # print('Candidate control string:', cand)
    #     # print the goal component of the prompt
    #     # print('Goal component of the prompt:', prompt.goal_str)
    #     cand_new = []
    #     for i in range(len(cand)):
    #         temp_cand = prompt.goal_str + ' ' + cand[i]
    #         cand_new.append(temp_cand)
    #     # cand_new = ' '.join(cand_new)
    #     # print('Candidate goal + control string:', cand_new)
    #     cand_tokens = tokenizer(cand_new).input_ids
        
    #     # Modified on 10/30/2025
    #     # if tokenizer.bos_token_id in cand_tokens:
    #     #     cand_tokens = cand_tokens[1:]  # Remove BOS token if present
    #     # cand_length = len(cand_tokens)
    #     cand_lengths = []
    #     for toks in cand_tokens_list:
    #         if hasattr(tokenizer, "bos_token_id") and toks and toks[0] == tokenizer.bos_token_id:
    #             toks = toks[1:]
    #         cand_lengths.append(len(toks))
        
    #     print('Candidate tokens length:', cand_length)
        
    #     if cand_length == 0:
    #         return torch.zeros(logits.shape[0], device=logits.device)
        
    #     # Find the position where the control suffix starts in the full sequence
    #     # This depends on your prompt structure - you may need to adjust this
    #     # control_start_idx = prompt._control_slice.start if hasattr(prompt, '_control_slice') else -cand_length --> this is the original code
    #     control_start_idx = 0 # take the prompt and the control string as a whole
    #     control_end_idx = control_start_idx + cand_length
        
    #     # Extract logits and target tokens for the control suffix
    #     if control_start_idx >= 0 and control_end_idx <= logits.shape[1]:
    #         # Logits for predicting the control tokens (shifted by 1 for next-token prediction)
    #         control_logits = logits[:, control_start_idx:control_end_idx-1, :]  # [batch, seq_len-1, vocab]
    #         control_targets = ids[:, control_start_idx+1:control_end_idx]      # [batch, seq_len-1]
            
    #         # Calculate cross-entropy loss for the control tokens
    #         control_logits_flat = control_logits.reshape(-1, control_logits.size(-1))
    #         control_targets_flat = control_targets.reshape(-1)
            
    #         # Calculate per-token cross-entropy
    #         ce_loss = F.cross_entropy(control_logits_flat, control_targets_flat, reduction='none')
    #         ce_loss = ce_loss.reshape(control_logits.shape[0], -1)  # [batch, seq_len-1]
            
    #         # Average cross-entropy over sequence length to get perplexity loss
    #         # We use the cross-entropy directly as the perplexity loss (since exp(ce) = perplexity)
    #         # This encourages lower cross-entropy, which corresponds to lower perplexity
    #         perplexity_loss = ce_loss.mean(dim=-1)  # [batch]
            
    #     else:
    #         # Fallback if we can't extract control tokens properly
    #         perplexity_loss = torch.zeros(logits.shape[0], device=logits.device)
        
    #     return perplexity_loss
    
    def calculate_perplexity_loss(self, prompt, logits, ids, cand):
        """
        Compute perplexity loss for the entire sequence (goal+control) per batch example.
        logits: Tensor [batch, seq_len, vocab]
        ids:    Tensor [batch, seq_len]
        cand:   list of candidate strings with batch size == logits.shape[0]
        """
        import torch
        import torch.nn.functional as F

        tokenizer = self.workers[0].tokenizer

        # For (optional) verification, reconstruct the input that produced logits/ids:
        # batch_strings = [prompt.goal_str + ' ' + c for c in cand]

        # For each example in the batch:
        batch_size = logits.size(0)
        ce_losses = []
        for i in range(batch_size):
            # Tokenize full string to get reference length and adjust for special tokens if needed
            full_str = prompt.goal_str + ' ' + cand[i]
            toks = tokenizer(full_str, add_special_tokens=True).input_ids  # +special tokens?
            if hasattr(tokenizer, "bos_token_id") and toks and toks[0] == tokenizer.bos_token_id:
                toks = toks[1:]
            seq_len = len(toks)

            # logits/ids might be longer than toks due to padding; clip to tokenized length
            if seq_len < 2:
                # Can't compute next-token loss on sequence <2
                ce_losses.append(torch.tensor(0.0, device=logits.device))
                continue

            l = seq_len
            # Prediction: logits for predicting tokens 1...l-1; targets: tokens 1...l-1
            # Usually, logits align with input tokens [0:l-1] -> predicting targets [1:l]
            example_logits = logits[i, 0:l-1, :]   # [seq_len-1, vocab]
            example_targets = ids[i, 1:l]          # [seq_len-1]

            # Cross-entropy loss (averaged over non-padded tokens)
            ce_loss = F.cross_entropy(example_logits, example_targets, reduction='mean')
            ce_losses.append(ce_loss)
        
        # Stack to tensor [batch]
        perplexity_loss = torch.stack(ce_losses)
        return perplexity_loss
