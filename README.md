# med-start
Example of workflow for medical AI training and testing


## **For Users**

## Instructions

1. Clone repository

2. Download and visualize the data using src/download_data.ipynb
















## **For repo Developers/Maintainers**

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


## Next Steps:
- Put the Dataset into the Datamodule so that there is only one file that needs changing with a different dataset.
    - Make sure that the Datamodule can be easily played around with in a Notebook for development purposes. Maybe best is the same as now where you copy/paste the Notebook Dataset in? Idk
- Should I rename ./data/ ? Rn it's confusing bc there's also ./src/data/ which is for source code...
- Achieve a fast_dev_run with the Hydra config system
- Get config/, models/, data/, and splits/.
    - I guess the config would need to specify the split used, but all splits would need to have the form so that the same Dataset/Datamodule can use them.
- Then, work on logging. We want to use MLFlow to log to, potentially, a common logging directory.
    - Use Hydra tags to automatically tag each run with the $USER that ran it. This will be important if a common logging directory is used.
- Logging framework should include local environment variables I think. Which local environment variables, though? Like their home directory or so? Maybe device=HPG or device=Local depending on HPG run vs. their local machine??
- Then, tests.
    - Would be nice to have tests verify that everything exists and instantiates properly. I think the tests they have in the template repo are pretty solid to get started.
    - Maybe first as an "extra" that is run before every run? Need to see how much time it will take to run lol

- Need to have some visualization stuff.
    - For visualizing results, for example. Some notebooks. But maybe some that's auto-run during eval.py?
- Need to have some built out pipeline for making the datamodule and making sure everything with the data is working as expected.
    - Like notebooks that pull from your config/config.yaml and src/data/my_datamodule.py so that they basically help you build it.


