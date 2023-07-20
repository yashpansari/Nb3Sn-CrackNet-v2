# Nb3Sn-CrackNet-v2
This is the declassified non-production version (and thus does not include all modules used in prod) of a codebase used in a research project at LBNL.
A package to identify, separate, extract, and analyse strands, filaments, and cracks, from stitched microscopic cross-sectional images of Nb3Sn Magnet Superconducting Coils. This algorithm was developed by me in a research project for the US Department of Energy, in coordination with research scientists and engineers at CERN and Lawrence Berkeley National Laboratory's ATAP division. The abstract, explanations of the algorithm, proof of results, test images, site of deployment, etc. are confidential information but cleared personnel could email me for more information on these.

# Deployment
In order to use the code, first ensure that you have a set of bright superconducter coil cross-section microscope images as tiff files, in a folder called "tiffs" in your home directory.  
  
Next, run the following commands in terminal or git-bash to download the code and set it up to run.  
  
```cd $HOME```   
```git clone https://github.com/yashpansari/Nb3Sn-CrackNet-v2.git```  
```mv ~/tiffs/ ~/Nb3Sn-CrackNet-v2/tiffs/```  
```cd ~/Nb3Sn-CrackNet-v2```  
  
After this, you must note 2 important things: Do you want to run the entire crack detection framework? and/or do you want to see the result plots? The answer to the first question can be called flag, and the second one can be called show. Next, just list the number of strands in each image. It is best to have 10 in each.  
  
The final command to run would be:  
```python3 main.py <flag> <show> <numbers>```  
For instance, if I had 5 images with 10 strands each, and I wanted to run the strand, filament, and crack detection but didnt want to see the plots, I would run:  
```python3 main.py True False 10 10 10 10 10```
