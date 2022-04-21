# Dimenet_modify
Dimenet++ pytorch implement &amp; 내마음대로 개조

본 모델은 Dimenet++ tensorflow version을 torch version으로 implement 시킨 모델입니다.

추가적으로 Multifidelity learning을 할 수 있도록 개발중입니다.

현재 master brunch는 state feature가 추가되지 않은 순수한 Dimenet++ model 입니다. 하지만 g09 data와 xtb data 모두 learning에 활용이 가능합니다.

state_plus brunch는 state feature가 layers.py에 추가된 Embedding 부분에서 state feature가 추가될 수 있도록 개조된 모델입니다. 
(state feature가 dimension 문제로 인해 molecule당 한번 들어가는 것이 아닌 bond별로 들어가 있는 상태로 최적화가 되어 있지 않은 상태입니다.)

Dimenet의 주요 개발 목표는 state feature를 추가하여 Multifidelity learning이 잘 수행될 수 있도록 모델을 개조하는 것과, learning time을 단축시키는 것입니다.
(현재 Multifidelity learning을 진행하였을 때 Megnet에서 감소하는 mae에 비해 Dimenet이 더 적게 mae가 감소하고 있는 상태로 
다시말해 state feature가 제대로 반영이 되지 않아 Multifidelity learning이 제대로 진행되고 있지 않은것으로 보입니다.)
