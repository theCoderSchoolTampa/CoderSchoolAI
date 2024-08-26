### CoderSchoolAI ###

The following is a full documentation write-up of the CoderSchoolAI library. For more information on how to use this library in an educational environment, feel free to contact the author of this library, Jonathan Koch, via:

Twitter: https://twitter.com/jonathanzkoch

Linkedin: https://www.linkedin.com/in/jonathanzkoch

Email: johnnykoch02@gmail.com

---

CoderSchoolAI is a comprehensive, open-source library designed to facilitate artificial intelligence learning and development, particularly focusing on reinforcement learning. The library offers a suite of easy-to-use tools and structures to create, train, and test reinforcement learning agents. These include, but are not limited to, various types of neural networks, state-of-the-art training algorithms, different environment shells, and buffer functionalities for experience replay.

The library's codebase is written in Python and integrates smoothly with the PyTorch framework, thereby allowing for optimal performance and flexibility. Its design makes it an excellent resource for both beginners learning reinforcement learning concepts and seasoned developers looking for a reliable and customizable toolkit.

The main components of CoderSchoolAI are:

- **Neural**: A sub-module that provides an assortment of neural network blocks such as InputBlock, ConvBlock, and OutputBlock, and a way to combine them into a full network (Net class). These tools allow for straightforward creation of various neural network architectures.

- **Environment**: This sub-module hosts different game environments that can be used for training agents, for instance, the SnakeEnvironment.

- **Agent**: This part of the library provides an abstract base class for developing different types of reinforcement learning agents.  

- **Training**: This sub-module includes implementation of reinforcement learning algorithms, such as Deep Q Learning and Q Learning, and functions to facilitate the training process of agents.

- **Utils**: This contains various types utils including replay buffers for experience replay in reinforcement learning algorithms, search utils, data utils and more!

Detailed usage and documentation of these components are provided in subsequent sections of this document.

Whether you are an AI enthusiast, a student, or a researcher, we hope you find this library to be an efficient and enjoyable tool in your journey through the exciting landscape of artificial intelligence. I am always interested in hearing about your experiences, suggestions, or any cool projects you've built using CoderSchoolAI. Reach out and let me know!

Copyright (c): 
This project is licensed under the terms of the MIT License. This means that you are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, subject to the following conditions:

The above copyright notice and this permission notice (including the next paragraph) shall be included in all copies or substantial portions of the software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

By using this library, you recognize that credit must be given to the authors of the library.

CoderSchoolAI is provided as a framework, without any future guarantees of improvement of any kind. For any queries, issues, or contributions, please refer to the project's [GitHub page](https://github.com/theCoderSchoolTampa/CoderSchoolAI.git).

---

Best, 

Jonathanzkoch

# CoderSchoolAI.Environment Documentation #

## Overview
This library provides an environment shell which is crucial for creating custom environments for Reinforcement Learning applications. It is based on the principles of OpenAI's Stable Baselines Gym environment.

## Shell Class
This class provides the basic structure of an environment. It handles the attributes, interactions, updates and provides utility functions needed for an environment. 

```python
class Shell:
    def __init__(self, target_fps: int, is_user_control: bool, resolution: Tuple[int, int], environment_name: str, verbose= False, console_only=False):
        """[User Implemented] Defines agents, environment vars, and state/action attributes. User must register attributes that they want tracked by the API."""
    
    def get_attribute(self, name) -> Union[
        Union[ObsAttribute, ActionAttribute], 
        Dict[str, Union[ObsAttribute, ActionAttribute]]]:
        """Retrieves attributes that are registered with the environment."""
    
    def get_observation(self, attributes=None) -> Dict[str, np.ndarray]:
        pass
    
    def register_attribute(self, attribute: Union[ObsAttribute, ActionAttribute]):
        pass
    
    def update_env_attributes(self):
        pass
    
    def update_attribute(self, name, new_data):
        """Updates the value of an attribute with an instance of `new_data`"""
    
    def reset(self, attributes=None) -> Union[Dict[str, ObsAttribute], ObsAttribute, np.ndarray]:
        """[User Implemented] Resets the environment and returns the new observation from the environment"""

    def step(self, 
        action: Union[int, np.ndarray, Dict[str, ActionAttribute]], 
        d_t:float, 
        attributes=None) -> Tuple[Union[Dict[str, ObsAttribute], ObsAttribute, np.ndarray], Union[int, float], Union[bool, np.ndarray]]:
        """[User Implemented] Steps the environment with `action` and returns `next_state, reward, done`"""

    def get_current_reward(
        self, 
        action: Union[int, np.ndarray, Dict[str, ActionAttribute]], 
        current_state: Union[Dict[str, ObsAttribute], ObsAttribute, np.ndarray]) -> Tuple[Union[int, float], bool]:
        """[User Implemented] returns the current reward given a `state` and `action`."""

    def update_env(self) -> None:
        pass
    
    def render_env(self):
        pass
    
    @staticmethod
    def static_render_env(env: 'Shell', *args, **kwargs):
        pass
```

### Methods
- `__init__(self, target_fps:int, is_user_control:bool, resolution:Tuple[int, int], environment_name:str, verbose:bool=False, console_only:bool=False)`: Initializes the Shell with various parameters such as target framerate, control type, screen resolution, environment name, verbosity, and mode of operation (console or GUI).

- `__getitem__(self, name)`: Gets the instance of the Attribute class for the specified attribute.

- `get_attribute(self, name) -> Union[Union[ObsAttribute, ActionAttribute], Dict[str, Union[ObsAttribute, ActionAttribute]]]`: Similar to `__getitem__`, but directly retrieves an attribute given its name.

- `get_observation(self, attributes=None) -> Dict[str, np.ndarray]`: Retrieves the current state of the environment in the form of a dictionary with numpy arrays.

- `register_attribute(self, attribute: Union[ObsAttribute, ActionAttribute])`: Registers a new attribute to the environment with a specified name and range.

- `update_env_attributes(self)`: Updates the Environment's attributes to match the current state of the Environment.

- `update_attribute(self, name, new_data)`: Allows updating the specified attribute with new data.

- `reset(self, attributes=None) -> Union[Dict[str, ObsAttribute], ObsAttribute, np.ndarray]`: Resets the environment and returns the initial state.

- `step(self, action: Union[int, np.ndarray, Dict[str, ActionAttribute]], d_t:float, attributes=None) -> Tuple[Union[Dict[str, ObsAttribute], ObsAttribute, np.ndarray], Union[int, float], Union[bool, np.ndarray]]`: Moves the environment forward by one step given

 a certain action. It returns the new observation, the reward and whether the episode is done.

- `get_current_reward(self, action: Union[int, np.ndarray, Dict[str, ActionAttribute]], current_state: Union[Dict[str, ObsAttribute], ObsAttribute, np.ndarray]) -> Tuple[Union[int, float], bool]`: Gets the reward for the specified action for a particular state. Returns the reward and the done flag for the environment.

- `update_env(self) -> None`: Updates the environment. If it's a PyGame Based environment, this method should call render to render the environment.

- `render_env(self)`: Should be implemented for PyGame based environments which need rendering to the screen for visualization.

- `static_render_env(env: 'Shell', *args, **kwargs)`: This is an unimplemented method intended for customization of the Rendering of the Environment.

## Attribute Class
The Attribute class represents an attribute of an environment, such as velocity or position. These attributes can be observation attributes (state of the environment) or action attributes (actions that can be performed in the environment). User registers Attributes with an environment for them to be tracked and updated.

```python
class Attribute:
    def __init__(self, name: str, space: SpaceType, update_func: Callable = None):
        pass
    def _get_shape(self):
        pass
    def update(self, data):
        pass
    def sample(self, distribution:np.ndarray=None):
        pass
```
### Methods
- `__init__(self, name:str, space:SpaceType, update_func:Callable = None)`: Initializes an attribute with a specified name, type, and an optional update function.

- `_get_shape(self)`: Determines the shape of the attribute data.

- `update(self, data)`: Updates the attribute data and its shape.

- `sample(self, distribution:np.ndarray=None)`: Samples the attribute data from a normal distribution if no distribution is provided. Otherwise, it samples from the provided distribution.

## ObsAttribute Class
The ObsAttribute class represents an observation attribute of an environment. It inherits from the Attribute class.

```python
class ObsAttribute(Attribute):
    def __init__(self, name: str, space: SpaceType, update_func: Callable = None):
        pass
```
### Methods
- `__init__(self, name:str, space:SpaceType, update_func:Callable = None)`: Initializes an observation attribute with a specified name, type, and an optional update function.

## ActionAttribute Class
The ActionAttribute class represents an action attribute of an environment. It inherits from the Attribute class.

```python
class ActionAttribute(Attribute):
    def __init__(self, name: str, space: SpaceType, update_func: Callable = None):
        pass
```
### Methods
- `__init__(self, name:str, space:SpaceType, update_func:Callable = None)`: Initializes an action attribute with a specified name, type, and an optional update function.

## Types
- `SpaceType`: A union of the different spaces that can be used for an attribute, including `BoxType`, `DiscreteType`, `MultiDiscreteType`, and `MultiBinaryType`.

Please note that the use of "type" and "specification" in the docstrings seems to be a misnomer. In this context, "type" and "specification" refer to the "space" of the attribute, which can be of the types mentioned in the `SpaceType` Union. 

For example, a Box space represents an n-dimensional box, so valid observations will be an array of 4 numbers for a Box(4) space. Similarly, a Discrete space is a set of non-negative numbers {0, 1, ..., n-1}. It is used for a fixed range of non-negative numbers. The spaces dictate the form and limits of the attributes.

## Agent Class
The Agent class represents an AI agent interacting with an environment. 

```python
class Agent:
    def __init__(self, is_user_control=False):
        pass
    def get_actions(self):
        pass
    def get_next_action(self, state):
        pass
    def update(self, state, action, next_state, reward):
        pass
```

### Methods
- `__init__(self, is_user_control=False)`: Initializes an agent with an optional flag indicating if the agent is under user control.

- `get_actions(self)`: Returns the list of actions that the agent can take in the current state of the environment. This method must be implemented in a subclass.

- `get_next_action(self, state)`: Returns the next action to be taken by the agent based on the current state of the environment. This method must be implemented in a subclass.

- `update(self, state, action, next_state, reward)`: Updates the agent's knowledge or parameters based on the observed state, action, next state, and reward from the environment. This method must be implemented in a subclass.

## ReplayBuffer
ReplayBuffer classes are used to store and retrieve the experiences of an agent during training. It helps in training the agent with a technique called Experience Replay.

There are two types of ReplayBuffer classes: BasicReplayBuffer and DictReplayBuffer.

### BasicReplayBuffer
```python
class BasicReplayBuffer(ReplayBuffer):
    def __init__(self, batch_size):
        pass
    def generate_batches(self):
        pass
    def store_memory(self, state, action, probs, vals, reward, done):
        pass
    def clear_memory(self):
        pass
```
#### Methods
- `__init__(self, batch_size)`: Initializes a replay buffer with a specified batch size.

- `generate_batches(self)`: Returns the experiences stored in the buffer in batches of the defined batch size. This method must be implemented in a subclass.

- `store_memory(self, state, action, probs, vals, reward, done)`: Stores an experience (state, action, probs, vals, reward, done) in the replay buffer.

- `clear_memory(self)`: Clears all the experiences stored in the replay buffer.

### DictReplayBuffer
```python
class DictReplayBuffer(ReplayBuffer):
    def __init__(self, batch_size):
        pass
    def generate_batches(self):
        pass
    def store_memory(self, state, action, probs, vals, reward, done):
        pass
    def clear_memory(self):
        pass
```
#### Methods
- `__init__(self, batch_size)`: Initializes a replay buffer with a specified batch size.

- `generate_batches(self)`: Returns the experiences stored in the buffer in batches of the defined batch size. This method must be implemented in a subclass.

- `store_memory(self, state, action, probs, vals, reward, done)`: Stores an experience (state, action, probs, vals, reward, done) in the replay buffer. In this case, state and action are stored as dictionaries.

- `clear_memory(self)`: Clears all the experiences stored in the replay buffer. It clears the memory for each state-action pair separately in this case.

## Example Usage

This section provides an example of how to use a custom environment class that extends the base `Shell` class provided by the `CoderSchoolAI` library. For the purposes of this demonstration, we will create a dummy environment and a dummy agent to illustrate the workflow.

Please note that this is a high-level example, and the specific methods and variables will depend on your particular environment and agents' specifics.

### Step 1: Define your Environment Class

Your environment should extend the `Shell` class. You need to implement methods such as `reset()`, `step()`, `get_current_reward()`, `update_env()`, and `render_env()`.

```python
class MyEnvironment(Shell):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize environment specific variables and 

    def reset(self, attributes=None):
        # Reset environment to the initial state
        pass

    def step(self, action, d_t, attributes=None):
        # Step the environment and return the new state, reward and done flag
        pass

    def get_current_reward(self, action, current_state):
        # Return the reward for the current state and action
        pass

    def update_env(self):
        # Update the environment's state
        pass

    def render_env(self):
        # Render the environment
        pass
```

### Step 2: Define your Agent Class

Your agent should have a method `get_next_action(state)` that takes in a state and returns an action. You may also need to implement other functions depending on your agent's learning algorithm.

```python
class MyAgent:
    def __init__(self, *args, **kwargs):
        # Initialize agent specific variables

    def get_next_action(self, state):
        # Decide the next action based on the current state
        pass
```

### Step 3: Run the Environment

With the environment and agent classes defined, you can now run your reinforcement learning experiment. Here is a simplified loop:

```python
my_env = MyEnvironment()  # Initialize your environment
my_agent = MyAgent()  # Initialize your agent

while True:  # Loop for each episode
    state = my_env.reset()  # Reset the environment
    done = False
    while not done:  # Loop for each step of the episode
        action = my_agent.get_next_action(state)  # Get an action from the agent
        state, reward, done = my_env.step(action)  # Step the environment
        my_env.update_env()  # Update the environment
        my_env.render_env()  # Render the environment (if applicable)
```

This code provides a generic example of how to use your custom environment and agent. Depending on the specifics of your classes and the reinforcement learning algorithm you're using, you may need to modify this template to fit your needs.

Remember to adhere to the rules of your environment and the agent's decision-making algorithm. Reinforcement learning can be complex, and small errors can have large impacts on your results.

# CoderSchoolAI.Neural Documentation #

## Overview
This library provides essential components to create and manipulate Deep Neural Networks for use with the CoderSchoolAI.Blocks Modules. Here are the main classes:

## Net Class
This class builds the network structure using the provided Blocks (InputBlock, LinearBlock, JoinBlock, OutputBlock, etc.). 

```python
class Net (nn.Module):
    def __init__(self, network_name:str= "basic_net", is_dict_network=False, device: th.device = th.device('cpu')):
        pass
    def add_block(self, block: Block):
        pass
    def compile(self):
        pass
    def forward(self, x: th.Tensor):
        pass
    def copy(self):
        pass
```

### Methods
- `__init__(self, network_name:str= "basic_net", is_dict_network=False, device: th.device = th.device('cpu'))`: This initializes the Net. It can be named for organization, declared as a dictionary network, and assigned a specific device for computation.

- `add_block(self, block: Block)`: Adds a Block connection to the network. The type of Block is based on the input.

- `compile(self)`: Once all blocks have been added, this function will compile the network and prepare it for training.

- `forward(self, x: th.Tensor)`: This function is called during training to pass the input data through the network.

- `copy(self)`: This function copies the current network, useful for creating multiple instances of the same network.


## InputBlock Class

The InputBlock connects all input data to a corresponding block.

```python
class InputBlock(Block):
    def __init__(self,
                 in_attribute:Union[Attribute, Dict[str, Attribute]], 
                 is_module_dict:bool=False, 
                 device: th.device = th.device('cpu')):
        pass
    def forward(self, x) -> th.Tensor:
        pass
```

### Methods

- `__init__(self, in_attribute:Union[Attribute, Dict[str, Attribute]], is_module_dict:bool=False, device: th.device = th.device('cpu'))`: Initializes the InputBlock with attributes, declares if it is a module dictionary, and assigns the computational device.

- `forward(self, x) -> th.Tensor`: This function is called during training to pass the input data through the block.


## JoinBlock Class

The JoinBlock merges the output from multiple preceding blocks into a single output.

```python
class JoinBlock(Block):
    def __init__(self,
                 join_size: int,
                 device: th.device = th.device('cpu'), 
                 activation: Optional[Callable] = None):
        pass
```

### Methods

- `__init__(self, join_size: int, device: th.device = th.device('cpu'), activation: Optional[Callable] = None)`: Initializes the JoinBlock with a specified join size, the computational device, and an optional activation function.


## LinearBlock Class

The LinearBlock creates a fully connected sequence of linear layers.

```python
class LinearBlock(Block):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 num_hidden_layers: int = 3,
                 hidden_size: Union[int, List[int]] = 128,
                 dropout: Optional[float] = None,
                 activation: Optional[Callable] = None,
                 device: th.device = th.device('cpu')):
        pass
    def join_block(self, block: Block, key:str = None):
        pass
    def regenerate_network(self):
        pass
```

### Methods

- `__init__(self, input_size: int, output_size: int, num_hidden_layers: int = 

3, hidden_size: Union[int, List[int]] = 128, dropout: Optional[float] = None, activation: Optional[Callable] = None, device: th.device = th.device('cpu'))`: Initializes the LinearBlock with input size, output size, number of hidden layers, hidden layer sizes, an optional dropout rate, an optional activation function, and the computational device.

- `join_block(self, block: Block, key:str = None)`: This function joins this block with another block. The key is optional and is required when joining with dictionary networks.

- `regenerate_network(self)`: This function corrects/builds a network from the internal state/structure of the block.


## OutputBlock Class

The OutputBlock takes the final input from the network and processes it into a usable output.

```python
class OutputBlock(Block):
    def __init__(self,
                 input_size: int,
                 num_classes: int,
                 activation: Callable = Identity,
                 device: th.device = th.device('cpu')):
        pass
```

### Methods

- `__init__(self, input_size: int, num_classes: int, activation: Callable = Identity, device: th.device = th.device('cpu'))`: Initializes the OutputBlock with the size of the final input, number of output classes, an activation function, and the computational device.


## Example Usage

Here is an example of constructing a network using this library for the SnakeEnv:

```python
from CoderSchoolAI.Environments.CoderSchoolEnvironments.SnakeEnvironment import *
from CoderSchoolAI.Neural.Blocks import *
from CoderSchoolAI.Neural.Net import *
import torch as th

snake_env = SnakeEnv(width=16, height=16)

input_block = InputBlock(snake_env.get_attribute("game_state"), False)
conv_block = ConvBlock(input_block.in_attribute.space.shape, 1, 3)
out_block = OutputBlock(conv_block.output_size, len(snake_env.snake_agent.get_actions()))
net = Net()
net.add_block(input_block)
net.add_block(conv_block)
net.add_block(out_block)
net.compile()
input_sample = snake_env.get_attribute("game_state").sample()
output_test = net(input_sample)
copy_net = net.copy()
output_copy_test = copy_net(input_sample)
```

# CoderSchoolAI.Training Documentation #

## Overview

The module provides functionalities for training reinforcement learning models. It includes methods for Deep Q-Learning, Q-Learning, and utility functions for Q-Learning such as loading and saving Q-Tables. The main functions in this module revolve around training a given agent in a specific environment, as well as updating the parameters of the agent as it interacts with the environment.

### `Datasets`

The `CoderSchoolAI.Training.Datasets` module provides several pre-implemented datasets that are ready to use:

- MNIST
- CIFAR-10
- Fashion-MNIST
- CIFAR-100
- ImageNet

First, you must create a Network for the dataset:
```python
from CoderSchoolAI.Environments.CoderSchoolEnvironments.SnakeEnvironment import *
from CoderSchoolAI.Environments.Attributes import *
from CoderSchoolAI.Neural.Blocks import *
from CoderSchoolAI.Neural.Net import *
from CoderSchoolAI.Training.Datasets import train_on_dataset, MNISTDataset

device = "cuda"

# the image is a 28x28 grid of pixels ranging between 0 and 1, this is where our shape comes from.
# the input block will take our image as input and output to the rest of the netwrok
image = ObsAttribute(name="img", space=BoxType(-1, 1, shape=(1, 28, 28)))
input_block = InputBlock(in_attribute=image, device=device)

# Define the ConvBlock which acts as a convolutional layer for processing the data. A channel size of 1 defines the image as a black and white image, and the depth of 4 creates 4 layers of convolutions deep for the image.
conv_block = ConvBlock(input_shape=input_block.in_attribute.space.shape, num_channels=1, depth=4, device=device)

# the OutputBlock will map from the convolutional block to an output for each of our 10 digits.
# The num_classes corresponds to the number of possible digits our network might see.
out_block = OutputBlock(input_size=conv_block.output_size, num_classes=10, device=device)

# Initializes the network and add the blocks
# By this point we have a network that can see the image, perform convolutions to the image and then output to the 10 digits in the MNIST dataset:
net = Net(device='cuda')

for block in (input_block, conv_block, out_block):
    net.add_block(block)

net.compile()
```

To use these datasets, you can import them directly from the module:

```python
from CoderSchoolAI.Training.Datasets import MNISTDataset, CIFAR10Dataset, FashionMNISTDataset, CIFAR100Dataset, ImageNetDataset

# Create a dataset instance
mnist_dataset = MNISTDataset()
cifar10_dataset = CIFAR10Dataset()
fashion_mnist_dataset = FashionMNISTDataset()
cifar100_dataset = CIFAR100Dataset()
imagenet_dataset = ImageNetDataset(root='/path/to/imagenet')  # ImageNet requires your machine to have a root path configured with this dataset in ImageNet format
```

You can then use these datasets with the train_on_dataset function:

```python
from CoderSchoolAI.Training.Datasets import train_on_dataset

train_on_dataset(net, mnist_dataset, epochs=5)
```

### Custom Transforms
Each dataset comes with a default transform, but you can provide your own custom transform:

```python
from torchvision import transforms

custom_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist_dataset = MNISTDataset(transform=custom_transform)
```

### Creating Custom Datasets
Creating datasets is useful to interface to the train function. To create a custom dataset, you need to inherit from the Dataset base class and implement the _load_data and _default_transform methods:

```python
from CoderSchoolAI.Training.Datasets import Dataset, DatasetOrigin
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, transform=None):
        super().__init__(DatasetOrigin.CUSTOM, transform)

    def _load_data(self):
        # Load your dataset here
        # Set self.trainset, self.testset, and self.num_classes
        pass

    def _default_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            # Add any other necessary transforms
        ])
```


## `deep_q_learning()` Function

This function implements the Deep Q-Learning algorithm which is used to train an agent in a given environment. Deep Q-Learning is a method for learning how to play a game by using a deep neural network to approximate the optimal action-value function. This function also uses a replay buffer to store transitions for experience replay.

```python
def deep_q_learning(
    agent: Agent,
    environment: Shell,
    q_network: Net,
    target_q_network: Net,
    buffer: Union[BasicReplayBuffer, DictReplayBuffer],
    num_episodes:int = 1000,
    max_steps_per_episode: int = 100,
    gamma: float = 0.99,
    update_target_every: int = 10,
    batch_size: int = 32,
    epsilon=0.9,
    epsilon_decay=0.997,
    stop_epsilon=0.01,
    alpha=0.01,
    attributes: Union[str, Tuple[str]] = None,
    optimizer= None,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
    fps: int = 120,
    ) -> None:
    pass
```

## `PPO()` Function

This function implements the Proximal Policy Optimization (PPO) algorithm, which is used to train an agent in a given environment. PPO is an on-policy algorithm that aims to compute an update at each step that minimizes the cost function while ensuring the deviation from the previous policy is relatively small.

```python
def PPO(
    agent: Agent,
    environment: Union[Shell, List[Shell]],
    actor_critic_net: ActorCritic,
    buffer: Union[BasicReplayBuffer, DictReplayBuffer],
    num_episodes: int = 1000,
    max_steps_per_episode: int = 100,
    gamma: float = 0.99,
    batch_size: int = 32,
    clip_epsilon: float = 0.2,
    alpha: float = 0.001,
    epsilon: float = 0.0001,
    entropy_coef: float = 0.005,
    critic_coef: float = 0.8,
    attributes: Union[str, Tuple[str]] = None,
    optimizer: Optional[th.optim.Optimizer] = None,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ppo_epochs: int = 4,
    minibatch_size: int = 16,
    fps: int = 120,
    max_grad_norm: float = 1.0,
    reward_norm_coef: float = 1.0,
    reward_normalization: bool = True,
    running_reward_std: float = 1.0,
    log_frequency: int = 10,
    logging_callback: Optional[Callable] = None,
) -> None:
    pass
```

### Parameters:

- `agent` (Agent): The agent that interacts with the environment.
- `environment` (Union[Shell, List[Shell]]): The environment(s) in which the agent operates.
- `actor_critic_net` (ActorCritic): The neural network that serves as both the actor and the critic.
- `buffer` (Union[BasicReplayBuffer, DictReplayBuffer]): The replay buffer to store experiences.
- `num_episodes` (int): The total number of episodes to train for.
- `max_steps_per_episode` (int): The maximum number of steps allowed in each episode.
- `gamma` (float): The discount factor for future rewards.
- `batch_size` (int): The size of each batch used for training.
- `clip_epsilon` (float): The clipping parameter for PPO.
- `alpha` (float): The learning rate for the optimizer.
- `epsilon` (float): A small value to prevent division by zero.
- `entropy_coef` (float): The coefficient for the entropy term in the loss function.
- `critic_coef` (float): The coefficient for the critic loss in the total loss function.
- `attributes` (Union[str, Tuple[str]]): The attributes to be used by the network.
- `optimizer` (Optional[th.optim.Optimizer]): The optimizer to use for training.
- `optimizer_kwargs` (Optional[Dict[str, Any]]): Additional keyword arguments for the optimizer.
- `ppo_epochs` (int): The number of epochs to train on each batch of data.
- `minibatch_size` (int): The size of mini-batches used in each PPO epoch.
- `fps` (int): The frames per second for environment execution.
- `max_grad_norm` (float): The maximum norm for gradient clipping.
- `reward_norm_coef` (float): The coefficient for reward normalization.
- `reward_normalization` (bool): Whether to use reward normalization.
- `running_reward_std` (float): The running standard deviation of rewards for normalization.
- `log_frequency` (int): How often to log training progress.
- `logging_callback` (Optional[Callable]): An optional callback function for custom logging.


### Description:

The PPO algorithm improves upon previous policy gradient methods by using a surrogate objective function and clipped probability ratios. This approach helps to ensure stable learning by preventing excessively large policy updates.

Key features of this implementation include:

1. Support for both single and multiple environments.
2. Use of an actor-critic architecture for improved stability.
3. Clipping of the surrogate objective to prevent too large policy updates.
4. Separate coefficients for entropy and critic loss to balance exploration and value estimation.
5. Option for reward normalization to handle varying scales of rewards across different environments.
6. Gradient clipping to prevent exploding gradients.
7. Flexible logging options including a custom callback for detailed monitoring.

### `QLearning()` Class

This class provides functionalities for Q-Learning, a model-free reinforcement learning algorithm which is used to find an optimal action-selection policy using a Q function. The class provides methods to choose an action based on the current state, update the Q-Table based on the state and reward received, and save/load the Q-Table for persistence.

```python
class QLearning:
    def __init__(self, actions, alpha=0.5, gamma=0.9, epsilon=0.9, epsilon_decay=0.995, stop_epsilon=0.01):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.stop_epsilon = stop_epsilon
        self.actions = actions
        self.q_table = defaultdict(FloatDict)

    def choose_action(self, state):
        pass

    def update_q_table(self, state, action, reward, next_state):
        pass
        
    def save_q_table(self, file_name):
        pass

    def load_q_table(self, file_name):
        pass
```

## Example Usage

Here is an example of using these functionalities in the context of training a reinforcement learning agent to play a game:

```python
from CoderSchoolAI.Agent import Agent
from CoderSchoolAI.Environments.Shell import Shell
from CoderSchoolAI.Neural.Net import Net
from CoderSchoolAI.Training import deep_q_learning, QLearning
from CoderSchoolAI.Buffer import BasicReplayBuffer

# Initialization
agent = Agent()
environment = Shell()
q_network = Net()
target_q_network = Net()
buffer = BasicReplayBuffer()

# Deep Q Learning
deep_q_learning(
    agent,
    environment,
    q_network,
    target_q_network,
    buffer,
    num_episodes=1000,
    max_steps_per_episode=100,
    gamma=0.99,
    update_target_every=10,
    batch_size=32,
    epsilon=0.9,
    epsilon_decay=0.997,
    stop_epsilon=0.01,
    alpha=0.01,
    fps=

120
)

# Q Learning
q_learner = QLearning(agent.get_actions())
state = environment.get_state()
action = q_learner.choose_action(state)
reward, next_state = environment.step(action)
q_learner.update_q_table(state, action, reward, next_state)
q_learner.save_q_table('q_table.json')
q_learner.load_q_table('q_table.json')
```

