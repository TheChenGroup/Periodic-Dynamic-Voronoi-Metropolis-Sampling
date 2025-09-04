# Periodic Dynamic Voronoi Metropolis Sampling (PDVMS)



This project implements the Periodic Dynamic Voronoi Metropolis Sampling (PDVMS) algorithm. This method is designed to extract classical electron configurations from wavefunctions, offering a novel particle-based perspective on many-body electronic structures.Here, using the  `DeepSolid` wavefunction as an example, PDVMS can analyze wavefunctions obtained from any method. 



## Environment Setup and Installation



We strongly recommend using `conda` to manage the Python environment for this project to ensure the compatibility of all required packages.

1. **Create a Conda Environment:**

   Based on the requirements of Deepsolid, we suggest creating a new conda environment with the following commands. The Python version we are using is 3.8.19.

   ```
   conda create -n pdvms python=3.8.19
   conda activate pdvms
   ```

   

2. **Download DeepSolid**: The core dependency of this project is the `Deepsolid` library. Please clone it from the official GitHub repository:

   ```
   git clone https://github.com/bytedance/DeepSolid.git
   ```

   

3. **Install Dependencies:**

   After activating the pdvms environment, use pip to install the necessary packages. The recommended list of dependencies is in `requirements.txt`. You can use the following command to install the required dependencies.

   

   ```
   pip install -r requirements.txt
   cd DeepSolid
   pip install -e .
   ```

   

   *Note: The installation of `jax` and `jaxlib` may need to be adjusted based on your operating system and whether you are using a GPU. Please refer to the [official JAX installation guide](https://www.google.com/search?q=https://github.com/google/jax%23installation&authuser=3) for details. The versions we use are `jax==0.2.26`, `jaxlib==0.1.75+cuda11.cudnn82`. You can find the relevant versions at `https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`. Alternatively, you can use the command below to download the corresponding version.*

   

   ```	
   pip install jaxlib==0.1.75+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   ```

   



## Contributing



We welcome contributions of all forms! If you find any bugs or have suggestions for improvement, please feel free to submit Issues or Pull Requests.






