**Status:** Maintenance (expect bug fixes and minor updates)

## This is a fork of the OpenAI Spinning Up in the Deep RL repository.
The original repository can be found [here](https://github.com/openai/spinningup)
This fork is intended to be an up-to-date version of the original repository, with the following changes:

The code has been updated to be compatible with Pytorch version 1.7.1
we updated readme files and other documentation to be more accessible for beginners
we also provided a step-by-step guide for downloading and installing Spinningup on Windows(11) Without using a Linux subsystem.
It removes the Windows community's barrier to diving into the Deep RL research platform. 

## Installation

This is a step-by-step guide for running spinningup on Windows:

## Download & Installation:
**Step 1:** Download  and Install [ Anaconda ](https://www.anaconda.com/download) or [ Miniconda ](https://docs.conda.io/en/latest/miniconda.html) for Windows.

**Step 2:** 
- Download Microsoft c++ build tools from [ here ](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- While Installing, select  _**Desktop development with C++**_ : 
![image](https://github.com/openai/spinningup/assets/78081958/2e8fd06f-2a5c-48f8-b145-b23ccdbd161e)

**Step 2:** 
- Open Anaconda prompt 
- Create "spinningup" environment by executing the command `conda create -n spinningup python=3.6`. 
- Now activate "spinningup" environment by executing the command `conda activate spinningup`

**Step 4:**
- Ensure that you have the "git" installed. If not, you can download it from [here](https://gitforwindows.org/)
- Clone the "spinningup" repository using the command: `git clone https://github.com/openai/spinningup.git`
- Once the cloning process is complete, navigate to the directory "Spinningup" and locate the "setup.py" file. Proceed to modify the Torch version to 1.7.1 within that file as shown below:
![image](https://github.com/openai/spinningup/assets/78081958/642086a5-4264-47e6-b6ab-9ef931371ab4)

**Step 5:**
- Install **_Swig_** by running command `pip install swig`
- Install openCV-Python by running command `pip install opencv-python==4.1.2.30`
- Install mpi4py by running command `Conda install -c conda-forge mpi4py`

**Step 6:**
- Navigate to the "Spinningup" directory using the command: `cd spinningup`.
- Now run command `pip install -e .` 

## Check Your Install:
- Follow the spinningup tutorial to check your installation from [here](https://spinningup.openai.com/en/latest/user/installation.html#check-your-install)

you may come across error while plotting the results using the given command 
<img width="339" alt="image" src="https://github.com/openai/spinningup/assets/78081958/e2d31883-eeab-4775-bd2c-4b65c44430ec">

**Error:** 
```
Plotting from...
==================================================


==================================================
Traceback (most recent call last):
  File "C:\Users\project\spinningup\spinup\utils\plot.py", line 233, in <module>
    main()
  File "C:\Users\project\spinningup\spinup\utils\plot.py", line 230, in main
    estimator=args.est)
  File "C:\Users\project\spinningup\spinup\utils\plot.py", line 162, in make_plots
    plot_data(data, xaxis=xaxis, value=value, condition=condition, smooth=smooth, estimator=estimator)
  File "C:\Users\project\spinningup\spinup\utils\plot.py", line 31, in plot_data
    data = pd.concat(data, ignore_index=True)
  File "C:\Users\project\AppData\Local\anaconda3\envs\spinningup\lib\site-packages\pandas\core\reshape\concat.py", line 284, in concat
    sort=sort,
  File "C:\Users\project\AppData\Local\anaconda3\envs\spinningup\lib\site-packages\pandas\core\reshape\concat.py", line 331, in _init_
    raise ValueError("No objects to concatenate")
ValueError: No objects to concatenate
Traceback (most recent call last):
  File "C:\Users\project\AppData\Local\anaconda3\envs\spinningup\lib\runpy.py", line 193, in _run_module_as_main
    "_main_", mod_spec)
  File "C:\Users\project\AppData\Local\anaconda3\envs\spinningup\lib\runpy.py", line 85, in _run_code
    exec(code, run_globals)
  File "C:\Users\project\spinningup\spinup\run.py", line 243, in <module>
    subprocess.check_call(args, env=os.environ)
  File "C:\Users\project\AppData\Local\anaconda3\envs\spinningup\lib\subprocess.py", line 311, in check_call
    raise CalledProcessError(retcode, cmd)
subprocess.CalledProcessError: Command '['C:\\Users\\project\\AppData\\Local\\anaconda3\\envs\\spinningup\\python.exe', 'C:\\Users\\project\\spinningup\\spinup\\utils\\plot.py', 'data/installtest/installtest_s0']' returned non-zero exit status 1.
```
**Solution:**
This can be due to the path error your system would be facing. You can update the command by editing the path of `installest_s0`. you can just go to the folder and can copy the path:
```
python -m spinup.run plot C:\Users\project\spinningup\data\installtest\installtest_s0
```

**Note:** The installation has been done and run over Windows 11.


Welcome to Spinning Up in Deep RL! 
==================================

This is an educational resource produced by OpenAI that makes it easier to learn about deep reinforcement learning (deep RL).

For the unfamiliar: [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) (RL) is a machine learning approach for teaching agents how to solve tasks by trial and error. Deep RL refers to the combination of RL with [deep learning](http://ufldl.stanford.edu/tutorial/).

This module contains a variety of helpful resources, including:

- a short [introduction](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) to RL terminology, kinds of algorithms, and basic theory,
- an [essay](https://spinningup.openai.com/en/latest/spinningup/spinningup.html) about how to grow into an RL research role,
- a [curated list](https://spinningup.openai.com/en/latest/spinningup/keypapers.html) of important papers organized by topic,
- a well-documented [code repo](https://github.com/openai/spinningup) of short, standalone implementations of key algorithms,
- and a few [exercises](https://spinningup.openai.com/en/latest/spinningup/exercises.html) to serve as warm-ups.

Get started at [spinningup.openai.com](https://spinningup.openai.com)!


Citing Spinning Up
------------------

If you reference or use Spinning Up in your research, please cite:

```
@article{SpinningUp2018,
    author = {Achiam, Joshua},
    title = {{Spinning Up in Deep Reinforcement Learning}},
    year = {2018}
}
```
