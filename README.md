# med-start
Example of workflow for medical AI training and testing


## Constraints

- Easy to run for biomedical computer vision tasks
- Allows experiment tracking in free fashion
- Works well with UF HiPerGator supercomputer
- Easy to change config and to track config of past runs
- Good set of validation/tests for ease of reporting in publications etc.

### Stretch design constraints

- Make it easy to do CI/CD for updating and deployment of models. For now, this mostly means having good validation/testing setup and good organization of model versions.

## Paradigms/Ideas

- Use MLFlow for experiment tracking and run a self-hosted server as a part of the sbatch script when submitted to HPG SLURM. View the runs on local machine using SSH port-forwarding when you want to view. This means you can't access, say, on your phone browser like you can with WandB company hosting but I think this is a good compromise. I have gotten the MLFlow dashboard on my local machine through HiPerGator via SSH port-forwarding.
- PyTorch Lightning to eliminate as much boilerplate code as posible.
- Possibly use MONAI library for their well-built ecosystem. I don't know how to work it super well yet, though.
- Config using YAML files somehow. I am playing around with Lightning CLI to make it easy to switch between train and test but I'm not sure I like how abstracted away it feels.
