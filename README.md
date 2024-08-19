Requires godot python addon/asset/module (too big for this repo, install it through asset library)  
![flappy-AI](https://github.com/user-attachments/assets/4d5bef1e-7a6b-42b7-b385-6b7d9a7f0bc7)  
highscore: 99  
deep Q reinforcement learning  
4 inputs - bird y velocity, y distances to bottom and top pipe, x distance to right edge pipe  
2 hidden layers, size 80 and 40  
learning on random batches of 64 states every 4 frames, with game running on 20-30 fps, 3x sped up  

