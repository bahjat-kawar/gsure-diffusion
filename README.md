# GSURE-Based Diffusion Model Training with Corrupted Data
<a href="https://bahjat-kawar.github.io/">Bahjat Kawar</a><sup>\*</sup>, <a href="https://github.com/noamelata">Noam Elata</a><sup>\*</sup>, <a href="https://tomer.net.technion.ac.il/">Tomer Michaeli</a>, and <a href="https://elad.cs.technion.ac.il/">Michael Elad</a>, Technion.<br />
<sup>\*</sup>Equal contribution.

<img src="figures\denoising_fig.png" alt="GSURE-Diffusion" style="width:800px;"/>

This code implements <a href="https://arxiv.org/abs/2305.13128">GSURE-Based Diffusion Model Training with Corrupted Data</a>. 

## Running the Experiments
Please refer to `environment.yml` for packages that can be used to run the code. 

### Data preperation

To use the datasets as defined in the code place the datasets in the following paths:
```bash
GSUREDiffusion
├── datasets # all dataset files
│   ├── celeba # all CelebA files
│   └── fmri # all FastMRI files
```
The CelebA might be downloaded automatically upon running the code, the MRI data needs to be downloaded from [FastMRI](https://fastmri.org/).
Run the data creation scripts to create the corrupt GSURE datasets.
```
python create_data.py
python create_data_mri.py
```

### Model Checkpoints
Coming soon...


### Training

The general command to train a model is as follows:
```
python train.py -c configs/<name of the config to be used>.json
```
The training dataset, architecture and hyperparameters are defined in the config JSON file.
The following commands train an oracle model and a GSURE model respectively for the CelebA dataset:
```
python train.py -c configs/celeba32_oracle.json
python train.py -c configs/celeba32_p02_s001.json
```
The following commands train an oracle model and a GSURE model respectively for the MRI dataset:
```
python train.py -c configs/mri_oracle.json
python train.py -c configs/mri_R4.json
```

A finer setting of hyperparameters may be achieved by changing the config JSON file.

For multi-GPU training or wandb integration please see the usage of `train.py`

### Sample Generation
To generate samples unconditionally from trained model:
```
python generate.py  --config <config path> 
                    --number <number of images to generate> 
                    --batch <batch size> 
                    --output <output path> 
                    --steps <number of steps> 
                    --ddim <use ddim, leave empty for DDPM>
                    --eta <ddim eta for ddim, default 0.0>
                    --model-path <path to saved model checkpoint>
```

## References and Acknowledgements
```BibTeX
@article{kawar2023gsure,
  title={GSURE-Based Diffusion Model Training with Corrupted Data},
  author={Kawar, Bahjat and Elata, Noam and Michaeli, Tomer and Elad, Michael},
  journal={arXiv preprint arXiv:2305.13128},
  year={2023}
}
```

Data used in the preparation of this article were obtained from the NYU fastMRI Initiative
database. As such, NYU fastMRI investigators provided data but did not participate
in analysis or writing of this report. A listing of NYU fastMRI investigators, subject to updates, can
be found at fastmri.med.nyu.edu. The primary goal of fastMRI is to test whether machine learning
can aid in the reconstruction of medical images.
```BibTeX
@misc{zbontar2018fastMRI,
    title={{fastMRI}: An Open Dataset and Benchmarks for Accelerated {MRI}},
    author={Jure Zbontar and Florian Knoll and Anuroop Sriram and Tullie Murrell and Zhengnan Huang and Matthew J. Muckley and Aaron Defazio and Ruben Stern and Patricia Johnson and Mary Bruno and Marc Parente and Krzysztof J. Geras and Joe Katsnelson and Hersh Chandarana and Zizhao Zhang and Michal Drozdzal and Adriana Romero and Michael Rabbat and Pascal Vincent and Nafissa Yakubova and James Pinkerton and Duo Wang and Erich Owens and C. Lawrence Zitnick and Michael P. Recht and Daniel K. Sodickson and Yvonne W. Lui},
    journal = {ArXiv e-prints},
    archivePrefix = "arXiv",
    eprint = {1811.08839},
    year={2018}
}
@misc{tibrewala2023fastmri,
  title={{FastMRI Prostate}: A Publicly Available, Biparametric {MRI} Dataset to Advance Machine Learning for Prostate Cancer Imaging},
  author={Tibrewala, Radhika and Dutt, Tarun and Tong, Angela and Ginocchio, Luke and Keerthivasan, Mahesh B and Baete, Steven H and Chopra, Sumit and Lui, Yvonne W and Sodickson, Daniel K and Chandarana, Hersh and Johnson, Patricia M},
  journal = {ArXiv e-prints},
  archivePrefix = "arXiv",
  eprint={2304.09254},
  year={2023}
}
```

This implementation is based on / inspired by:
- [Unofficial Implementation of Palette: Image-to-Image Diffusion Models by Pytorch](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models)
- [Guided-Diffusion](https://github.com/openai/guided-diffusion)

