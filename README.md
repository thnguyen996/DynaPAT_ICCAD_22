
# DynaPAT (ICCAD 2022) Implementation
This repository is the official implementation of DynaPAT paper. Details of the proposed methods are
described in:  <em>[Thai-Hoang Nguyen](https://thaihoang.org), Muhammad Imran, Joon-Sung Yang (2022). [DynaPAT: A Dynamic Pattern-Aware Encoding Technique for Robust MLC PCM-Based Deep Neural Networks.](https://doi.org/10.1145/3508352.3549400) ICCAD'22: Proceedings of the 41st IEEE/ACM International Conference on Computer-Aided Design.</em>

### Requirements

<!-- Requirements: --> 
Install required packages: 

```
$ pip install -r requirement.txt
```

### GETTING STARTED
<!-- GETTING STARTED--> 
  
1. Clone the repo ```git clone https://github.com/thnguyen996/DynaPAT_ICCAD_22.git```
2. Download pretrained weights and put it in checkpoint/ folder 
3. Run the code  ```python main.py --help``` to see all the options provided by the code

### Baseline
Run the following code to evaluate resistance drift on a baseline model 
```
$ python main.py --model [name of the model] --method baseline -gpu [GPU ID] --num_bits 8
```

### Proposed method (DynaPAT): 
Run state_encoding.py to generate state encoding for DynaPAT 

```
$ python state_encoding.py --model [Name of the model] --method proposed_method 
```

Run main.py to evaluate DynaPAT for a given state encoding
```
$ python main.py --method proposed_method --model [Name of the model] --gpu [GPU ID]  --num_bits 8
```

### Flipcy and Helmet 

For Flipcy and Helmet, run the main.py file with addtional options --encode first, this will generate
encodings for the 2 methods
```
$ python main.py --method ECP 
```

### Research papers citing

If you use this code for your research paper, please use the following citation:

```
@inproceedings{nguyen2022dynapat,
 author = {Nguyen, Thai-Hoang and Imran, Muhammad and Yang, Joon-Sung},
 booktitle = {Proceedings of the 41st IEEE/ACM International Conference on Computer-Aided Design},
 pages = {1--9},
 title = {DynaPAT: A Dynamic Pattern-Aware Encoding Technique for Robust MLC PCM-Based Deep Neural Networks},
 year = {2022}
}

```
