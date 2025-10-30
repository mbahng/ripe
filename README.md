# \[Name\]

   \[Abstract\]

   Some teaser figures here. 
   ![image](saved/fig/diagram.png)


   Setup conda env. 
   ```
   conda create -n env python=3.13 -y
   conda create -p ~/xt/condaenvs/env python=3.13 -y
   ```
   Install relevant packages. 
   ```
   pip install torch torchvision scikit-learn numpy matplotlib pyyaml tqdm ipython debugpy wandb scipy gdown kagglehub
   ```

   Run example script to train MLP on MNIST. 
   ```
   python main.py --cfg=setup/cfg/example_mlp.yml
   ```

# Some Design Choices

   Here are some quick rues. 
   1. The dataloader should be collated to return a dictionary rather than tuples. This makes it more flexible to accommodate datasets that may have one, two, or more inputs. 
   2. Flattening data before forwarding into, say an MLP, should be done in the forward function of the `nn.Module` object. 

## Multiple Models/Optimizers/Losses

   If you need multiple models, either in the form of a model that contains a smaller model (such as a multimodal Prototree that contains an image and genetic unimodal prototrees), or multiple separate models (like a generator and discriminator for GANs) then you should create multiple instances under the relevant section. The same applies for losses and optimizers. 

   For example, in a GAN, we can change it to something like this. 
   ```
   model: 
     name: "gan" 
     generator: 
       hidden_dim: [64, 128, 256, 512]
       activations: ["relu", "relu", "relu", "sigmoid"]
     discriminator: 
       hidden_dim: [512, 256, 128] 
       activations: ["leaky_relu", "leaky_relu", "sigmoid"]
   ```
   Since these are two separate models, we can return a tuple of them 
   ```
   case "gan": 
     generator = Generator(**cfg_model["generator"]) 
     discriminator = Discriminator(**cfg_model["discriminator"]) 
   ```
   Or we can put them in a wrapper class called `GAN` or something, and then just return that 
   ```
   case "gan": 
     gan = GAN(**cfg_model)
   ```


   For example, for a multimodal prototree, 
   ```
   model:
     name: "prototree"
   ```
   would be changed to something like 
   ```
   model: 
     name: "multimodal_prototree" 
     depth: 11 
     image: 
       name: "image_prototree" 
       latent_dim: 2048
     genetic: 
       name: "genetic_prototree" 
       latent_dim: 64
   ```
   So when you actually call `init_model()`, you would add an extra case argument where the input dictionary argument provides all the keyword arguments needed to construct the unimodal image and genetic prototrees. 
   ```
   case "mmptree": 
     print("Loading Model: Multimodal ProtoTree")
     model = MultimodalProtoTree(**cfg_model)
   ```

   If you have multiple models/losses/optimizers, these will all have to be passed as arguments to the `Trainer` class. 
   ```
   train_dl, val_dl, test_dl = dataset.init_dataloader(cfg["dataset"])
   model = run.model.init_model(cfg["run"]["model"]).to(device)
   loss = run.loss.init_loss(cfg["run"]["loss"])
   optimizer = run.optimizer.init_optimizer(cfg["run"]["optimizer"], model)

   trainer = run.Trainer(train_dl, val_dl, test_dl, model, loss, optimizer, device=device)
   ```
   Therefore, the constructor signature will change quite a bit depending on which model you're implementing. I still need to think of a better way to design this, perhaps using a Builder Pattern. idk tho

## Logging distributions, image samples, etc. 

   Sometimes, you may want to see how the model's distribution evolves. All of these should be done in the logger. 

   All metrics that are floating point values (e.g. loss values, regularization values) should be saved every epoch. 
   All data that are tensors should be saved periodically (e.g. model weights, gradients)

   






