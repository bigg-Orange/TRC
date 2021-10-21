### Dependencies

* Compatible with TensorFlow 1.x and Python 3.x.
* Dependencies can be installed using `requirements.txt`.


### Usage:

* After installing python dependencies from `requirements.txt`.
* TensorFlow (1.x) . 
* To 
  ```shell
  python time_proj.py -data_type yago -margin 10 -model MODEL_NAME -test_freq 25 -<other_optins> ...
  ```
*  Some of the important Available options include:
  ```shell
    '-data_type' default ='wiki_data', choices = ['yago','wiki_data','WNP','YGP'], help ='dataset to choose'
    '-quatity_factor' default = 0.6 ,help = 'q(in paper)'
	'-version',  default = 'large', choices = ['large','small'], help = 'data version to choose'
	'-test_freq', 	 default = 25,   	type=int, 	help='testing frequency'
	'-neg_sample', 	 default = 5,   	type=int, 	help='negative samples for training'
	'-gpu', 	 dest="gpu", 		default='1',			help='GPU to use'
	'-name', 	 dest="name", 		help='Name of the run'
	'-lr',	 dest="lr", 		default=0.0001,  type=float,	help='Learning rate'
	'-margin', 	 dest="margin", 	default=1,   	type=float, 	help='margin'
	'-batch', 	 dest="batch_size", 	default= 50000,   	type=int, 	help='Batch size'
	'-epoch', 	 dest="max_epochs", 	default= 1000,   	type=int, 	help='Max epochs'
	'-l2', 	 dest="l2", 		default=0.0, 	type=float, 	help='L2 regularization'
	'-seed', 	 dest="seed", 		default=1234, 	type=int, 	help='Seed for randomization'
	'-inp_dim',  dest="inp_dim", 	default = 128,   	type=int, 	help='')
	'-L1_flag',  dest="L1_flag", 	action='store_false',   	 	help='Hidden state dimension of FC layer'
  ```

### Evaluation: 
* After trainig start validation/test. Use the same model name and test frequency used at training as arguments for the following evalutation--
* For getting MR and hit@10 for head and tail prediction:
 ```shell
    python result_eval.py -model MODEL_NAME  -test_freq 25
 ```
* For getting MR and hit@10 for relation prediction:
```shell
   python result_eval_relation.py -model MODEL_NAME  -test_freq 25
```
