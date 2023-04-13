# Conditional Neural ODE Processes for Disease Progression Forecasting

Code for "**Conditional Neural ODE Processes for Individual Disease Progression Forecasting: A Case Study on COVID-19**"  (Submission to KDD 2023). 



Note: this will be continuously updated.



## Getting started

For development, we used `Python 3.9.13` and `PyTorch 1.12.1. First, install `PyTorch`
using the [official page](https://pytorch.org/) and then run the following command to install the required packages:

```bash
pip install -r requirements.txt
```

## Running the experiments

To run the experiments:

```bash
python CNDP_covid.py --lr 1e-4 --decay 0.95 --is_aug True --inputdim 385 --posweight 1.5  --varname CNDPtest --user_y0
```

## Datasets

For more details on the COVID-19 dataset, please refer to:
```bash
@article{dang2022exploring,
  title={Exploring longitudinal cough, breath, and voice data for COVID-19 progression prediction via sequential deep learning: model development and validation},
  author={Dang, Ting and Han, Jing and Xia, Tong and Spathis, Dimitris and Bondareva, Erika and Siegele-Brown, Chlo{\"e} and Chauhan, Jagmohan and Grammenos, Andreas and Hasthanasombat, Apinan and Floto, R Andres and others},
  journal={Journal of medical Internet research},
  volume={24},
  number={6},
  pages={e37004},
  year={2022},
  publisher={JMIR Publications Toronto, Canada}
}
```

## Credits

Our code relies to a great extent on the [Neural ODE Processes](https://github.com/crisbodnar/ndp) by Cristian Bodnar. 

