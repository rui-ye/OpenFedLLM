import torch
import copy
from trl import DPOTrainer
from .fed_local_sft import SCAFFOLD_Callback

def get_fed_local_dpo_trainer(script_args, fed_args, model, model_ref, tokenizer, training_args, local_dataset, global_dict, local_auxiliary, global_auxiliary):
    
    if fed_args.fed_alg == 'fedprox':
        trainer = DPOTrainerFedProx(
                            model=model,
                            ref_model=model_ref,
                            args=training_args,
                            beta=script_args.dpo_beta,
                            train_dataset=local_dataset,
                            tokenizer=tokenizer,
                            global_state=global_dict,
                            prox_mu=fed_args.prox_mu,
                            )
    elif fed_args.fed_alg == 'scaffold':
        trainer = DPOTrainerSCAFFOLD(
                            model=model,
                            ref_model=model_ref,
                            args=training_args,
                            beta=script_args.dpo_beta,
                            train_dataset=local_dataset,
                            tokenizer=tokenizer,
                            global_state=global_dict,
                            local_auxiliary=local_auxiliary,
                            global_auxiliary=global_auxiliary,
                            )
        trainer.add_callback(SCAFFOLD_Callback(trainer.correction, model))
    elif (fed_args.fed_alg in ['fedavg', 'fedavgm', 'fedadgrad', 'fedyogi', 'fedadam']) or (fed_args.fed_alg).startswith('local'):
        trainer = DPOTrainer(
                            model=model,
                            ref_model=model_ref,
                            args=training_args,
                            beta=script_args.dpo_beta,
                            train_dataset=local_dataset,
                            tokenizer=tokenizer,
                            )
    else:
        raise ValueError(f'Unsupported `fed_alg`: {fed_args.fed_alg}')
    return trainer

class DPOTrainerFedProx(DPOTrainer):
    def __init__(self, global_state, prox_mu, **kwargs):
        super(DPOTrainerFedProx, self).__init__(**kwargs)
        self.global_state = global_state
        self.mu = prox_mu
    
    def compute_loss(self, model, inputs, return_outputs=False):

        return_values = super(DPOTrainerFedProx, self).compute_loss(model, inputs, return_outputs=return_outputs)

        if return_outputs:
            loss, outputs = return_values
        else:
            loss = return_values

        # Apply FedProx Loss
        for name, param in model.named_parameters():
            name = name.replace(".default", "")     # TODO: May need changes. to accord with peft
            # only trainable parameters
            if not param.requires_grad:
                continue
            else:
                loss += self.mu / 2 * torch.norm(param - self.global_state[name]) ** 2

        return (loss, outputs) if return_outputs else loss
    
class DPOTrainerSCAFFOLD(DPOTrainer):
    def __init__(self, global_state, local_auxiliary, global_auxiliary, **kwargs):
        super(DPOTrainerSCAFFOLD, self).__init__(**kwargs)
        self.global_state = global_state
        self.local_auxiliary = local_auxiliary
        self.global_auxiliary = global_auxiliary
        self.correction = copy.deepcopy(local_auxiliary)

        for name in self.correction.keys():
            self.correction[name] = self.global_auxiliary[name] - self.local_auxiliary[name]
    
    def get_auxiliary_param(self):
        auxiliary_new_para = copy.deepcopy(self.local_auxiliary)
        auxiliary_delta_para = copy.deepcopy(self.local_auxiliary)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                else:
                    name = name.replace(".default", "")
                    auxiliary_new_para[name] = (self.global_state[name] - param) / (self.args.max_steps * self.args.learning_rate) - self.correction[name]
                    auxiliary_delta_para[name] = auxiliary_new_para[name] - self.local_auxiliary[name]

        return auxiliary_new_para, auxiliary_delta_para