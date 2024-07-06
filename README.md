(Simplified) diffusion model implementation based on the [DDPM](https://arxiv.org/abs/2006.11239) paper.

----------------------------------------------------------------------

Some sample inference examples from my local machine.

<img src="./results/result-0.jpg" width="196px"><img></br>
<img src="./results/result-6.jpg" width="196px"><img></br>
<img src="./results/result-7.jpg" width="196px"><img></br>


The current implementation is a simplified version of the [hugging face annotated diffusion blog](https://huggingface.co/blog/annotated-diffusion). Also thanks to this [beautiful survey paper](https://arxiv.org/pdf/2406.08929) for increasing my intuitive understanding of the topic.

The model is trained on the [MNIST](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) digits dataset. I will make a deeper model and try it with a more complex dataset later.

To run a trained model locally,  download the repo and run the following. This will run the model with trained weights (that I trained on my local CPU only machine in 1hr) & produce one sample.

    python3 -m venv . 
    pip3 install -r requirements.txt
    cd src/main
    python3 test.py
 
To train the model, you can run `python3 main.py` inside the `src/main` directory. The trained weights will be stored in the same directory. You can change `test.py` 
to infer with the new weights.

----------------------------------------------------

Some current simplifications:
- No MLP for time embedding, and fixed an error with the time embedding implementation of the hf blog.
- Time embedding passed initially only, not to every layer in UNet. This is probably worse for inference quality (todo: experiment) but gave me a simple model to play around with.
- Simpler UNet (todo: experiment with deeper/more-res-connection models)
- Simplified inference using Algo-2 of the paper.
- No attention layer.

The [cold diffusion paper](https://arxiv.org/abs/2208.09392) does suggest that some of the technicalities of original DDPM might not be required, so some of the simplifications above may not cause inference quality problems. (todo: experiment)




