# RRB
The official code for the paper: "**Injecting Knowledge in Data-driven Vehicle Trajectory Predictors**", Published in Transportation research part C, 2021. 
[**Webpage**](https://mohammadhossein-bahari.github.io/RRB/) , [**Paper**](https://arxiv.org/pdf/2103.04854.pdf)

&nbsp;

<img src="pull.PNG" alt="drawing" width="600"/>




## Abstract:
Vehicle trajectory prediction tasks have been commonly tackled from two distinct
perspectives: either with knowledge-driven methods or more recently with datadriven ones. On the one hand, we can explicitly implement domain-knowledge
or physical priors such as anticipating that vehicles will follow the middle of the
roads. While this perspective leads to feasible outputs, it has limited performance
due to the difficulty to hand-craft complex interactions in urban environments.
On the other hand, recent works use data-driven approaches which can learn
complex interactions from the data leading to superior performance. However,
generalization, i.e., having accurate predictions on unseen data, is an issue leading
to unrealistic outputs. In this paper, we propose to learn a "Realistic Residual
Block" (RRB), which effectively connects these two perspectives. Our RRB
takes any off-the-shelf knowledge-driven model and finds the required residuals
to add to the knowledge-aware trajectory. Our proposed method outputs realistic
predictions by confining the residual range and taking into account its uncertainty.
We also constrain our output with Model Predictive Control (MPC) to satisfy
kinematic constraints. Using a publicly available dataset, we show that our method
outperforms previous works in terms of accuracy and generalization to new scenes.


### Installation ###
```
virtualenv -p /usr/bin/python3.6 rrb_env
source rrb_env/bin/activate
pip install -e trajnetbaselines/
pip install -e trajnettools/
pip install -e trajnetdataset/
```

### Model Training ###
You can specify code parameters in the bash.sh file. To train the network, simply run:
```
bash run.sh
```
### Model Evaluation ###
```
cd trajnetbaselines
python -m trajnetbaselines.eval --model-add <add-to-model>
```
You can evaluate the pre-trained models available in this repo with commands like this: 
```
python -m trajnetbaselines.eval --model-add 'output/final_models/RRB/RRB_M_sceneGeneralization'
```

## Citation: 
```
@article{bahari2021injecting,
  title={Injecting Knowledge in Data-driven Vehicle Trajectory Predictors},
  author={Bahari, Mohammadhossein and Nejjar, Ismail and Alahi, Alexandre},
  journal={arXiv preprint arXiv:2103.04854},
  year={2021}
}
```
### contact:
mohammadhossein.bahari@epfl.ch
