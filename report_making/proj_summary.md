my aim of the project is to find x and y coordinates of components such that the center of gravity is near 0. 
all the objects are 2d. there are many contraints to be satisfied on the way. this is the timeline that was followed:
1st go to previous_work folder. inside that there is 1.ipynb go through each cell. the first cell was given by my guide that 
arranges componets such that cg is 0. it uses DE as the algorithm and it asks every sing from the user iteratively. 
the second cell is from where i started my work. i had to implement a border of custom size. no elements were allowed to enter 
that border space. no elements are allowed to overlap.
the third cell tells about clearance space of custom size where minimum that much distance should be maintained from one element 
to the neighbor one. no elements are allowed to overlap but clearance space can overlap with border.
the next cell is regarding insert holes. insert holes are either of 4mm or 6mm in diameter. based on the size of inserts there should be minimum distance to be maintained from one insert to insert of another component. if both inserts are 4mm then the distance to be maintained is 24mm. if one is 4mm and another is 6mm then the distance to be maintained is 30mm. if both are 6mm then the distance to be maintained is 36mm.
all the above rules apply to this one too. 5 different layouts were supposed to be made for each cell for diversity so that engineers can choose which works best for them. 
since inputting details is a very tedious work, i created a json file which is now renamed as layout1.json.
the file using_jason.ipynb takes details from layout1.jason and gives all outputs.
all this was made using DE

now previous work folder is done
next comes DE folder
now i created a full pipeline. this included circular objects too. you can see everything in de.py file. and then elements were put on both front and backside of the board. a nice visualization was implemented. 2 images were created

DE part is done now
to find a faster way to solve everything i used CMA-es as my next algorithm
now comes the cma-es folder. this also does the same work but the only different thing is the algorithm. u can go through the code in cma-es-local.py. all its output are stored in a cam-es_output directory. please ignore if the output folder name or the json file is different int he code

Next comes the ppo folder
here there is a readme file from which u can understand most of the things. the ipynb file is used to train the model. and the gen_from+inference.py is used to generate the outputs using the model present. there is a folder called datasets that contains valid 1049 layouts in .npz format. i dont know if they were used to train the model or not. all the generated layouts are saved in the ppo_outputs directory. 

all this was done. now i had to compare the capabilities of all 3 algorithms which was better in different aspects. there is a run_comparitive study that tells how everything is generated. the outputs are saved in comparision_output directory. one correct convergence trajecotry is done in the ipynb file. 

yesterday my sir gave me a new layout to try on. it is the folder called new_layout. layout2.json contains the details of the new layout. there are 3 folder called cma_op, de_op, ppo_op which contaions the respective outputs. 

i need you to take the ppt as refereal point for everything. the ppt has detialed information regarding everything. include the results part only from ppt. total 8 are there. 

