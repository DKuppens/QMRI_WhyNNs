# QMRI Hypothesis Testing Framework — Multi-Model Edition

This codebase tests hypotheses about why neural networks outperform least-squares
fitting in quantitative MRI parameter estimation. All hypothesis scripts now
support two QMRI models: IVIM, and DCE.

---

## Repository layout

```
├── backend/
│   ├── QMRI/
│   │   ├── model_registry.py       
│   │   ├── ivim_model.py          
│   │   ├── dce_model.py          
│   ├── NN/
│   │   ├── NN.py                  
│   │   ├── train.py           
│   │   ├── nn_variants.py     
│   │   └── prediction.py          
│   │── experiment_utils.py        
│   │─ figure_data.py        
│   └── plot_style.py        
│
└── hypotheses/
    ├── H1_local_minima.py
    ├── H1_loss_landscape.py
    ├── H2_loss_landscape.py
    ├── H2_noise_dominated_minimum.py
    ├── H3_data_prior.py
    ├── H4_pseudo_improvement.py

```

## Running hypothesis scripts

Every hypothesis script accepts a `--model` argument:

```bash
# IVIM 
python experiments/H1_local_minima.py

# DCE
python experiments/H1_local_minima.py --model DCE
```
