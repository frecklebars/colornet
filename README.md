# text-color-predictor  
I made this to get more familiar with how neural networks work.  
What the program does is randomly pick a color and, based on its training, decides whether white or black text looks better over that color. 
It's trained with a back propagation algorithm and as for the dataset, it picks a random color and decides whether the text should be black or white using this formula: if R * 0.299 + G * 0.587 + B * 0.114 is higher than 150, it's black, else it's white.  
  
It finally works oh man, that's important. I'm pretty proud.  
you can mess around with it [here](http://butteredtoast.ml/projects/colornet/index.html)  
### looks kinda like this:
![](https://github.com/frecklebars/text-color-predictor/blob/master/net.png)
