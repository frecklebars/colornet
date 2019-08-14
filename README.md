# text-color-predictor  
I made this to get more familiar with how neural networks work.  
What the program does is randomly pick a color and, based on its training, decides whether white or black text looks better over that color. 
It's trained with a back propagation algorithm and as for the dataset, it picks a random color and decides whether the text should be black or white using this formula: if R * 0.299 + G * 0.587 + B * 0.114 is higher than 150, it's black, else it's white.  
  
It finally works oh man, that's important. I'm pretty proud.  
  
### later update
I'm trying to make a js version too for the sake of running in a browser with a nice window and all. It's still a work in progress.
