import openai
import tiktoken
from dill.source import getsource
import re

def remove_comments(source):
    """Remove comment lines from a string containing Python source code."""
    # Remove multi-line comments (/* ... */)
    source = re.sub(r'/\*.*?\*/', '', source, flags=re.DOTALL)
    
    # Remove single-line comments (# ...) and inline comments (x = 1 # ...)
    lines = source.split('\n')
    filtered_lines = []
    for line in lines:
        line = re.sub(r'#.*', '', line)
        line = line.rstrip()
        if line:
            filtered_lines.append(line)
    return '\n'.join(filtered_lines)

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")

def query_gpt(openai_key : str, user_final_query : str, system_message : str=None, few_shot_examples=None):
    openai.api_key = openai_key
    MODEL = "gpt-4-0314" # "gpt-3.5-turbo-0301" # "gpt-4-0314"

    messages = []
    if system_message is not None:
        messages.append({"role": "system", "content": system_message})
    if few_shot_examples is not None:
        for example in few_shot_examples:
            messages.append({"role": "system", "name": "example_user", "content": example[0]})
            messages.append({"role": "system", "name": "example_assistant", "content": example[1]})
    messages.append({"role": "user", "content": user_final_query})
    

    print("messages:", messages)
    print("num_tokens_from_messages:", num_tokens_from_messages(messages))

    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature=0,
    )
    return response

def query_gpt_wrapper(openai_key : str, user_query : str, context_description : str=""):
    api_description = \
    "The generated loss class should have a function \
    forward(x, data_batch, agt_mask). x is a tensor representing the current trajectory with shape (B, N, T, 6) where B is the number of vehicles (consisting of vehicle 0 to B-1), N is the number of samples for each vechile, T is the number of timesteps (each represents 0.1s), and 6 represents the (x, y, vel, yaw, acc, yawvel) in corresponding agent coordinate of each vehicle. data_batch is a dictionary that can be used as parameter for relevant APIs. The function should return a loss for every sample of every vehicle with the shape (B, N) or return a loss for every sample with the shape (N). \
    Some concepts: \
    1. vehicles exist in a 2d-world which has a world coordinate. \
    2. each vehicle has its own agent coordinate in which its current position (x,y) is used as the center and its orientation (yaw) is the positive x axis facing ahead. A vehicle faces towards positive x-axis in its own agent coordinate and a vehicle's left direction is the positive y-axis in its own agent coordinate. \
    3. we use the word 'agent' and the word 'vehicle' interchangeably. \
    You can use PyTorch and the following APIs if needed amd you should not use other unseen functions: \
    1. transform_coord_agents_to_world(pos_pred, yaw_pred, data_batch). pos_pred is the predicted position trajectory in agent coordinate with shape (B, N, T, 2) and 2 correspond to (x, y). yaw_pred is the predicted yaw trajectory in agent coordinate with shape (B, N, T, 1). The function transform the predicted position and yaw from their agent coordinates into the world coordinate. The function returns position and yaw in the world coordinate with the shape (B, N, T, 2) and (B, N, T, 1). \
    2. transform_coord_world_to_agent_i(pos_pred_world, yaw_pred_world, data_batch, ind_k). pos_pred is the predicted position trajectory in world coordinate with shape (B, N, T, 2) and 2 represents (x, y). yaw_pred is the predicted yaw trajectory in world coordinate with shape (B, N, T, 1). data_batch is the dictionary mentioned before. ind_k is the index whose agent coordinate will be converted to. The function transform the predicted position and yaw from world coordinate to the agent i coordinate. The function returns position and yaw in the agent i coordinate with the shape (B, N, T, 2) and (B, N, T, 1). \
    3. select_agent_ind(x, i). x has shape (B, N, T, k) where k can be any positive integer and i is a non-negative integer representing the selected index. This function returns the slice of x with index i with shape (N, T, k). \
    4. get_current_lane_projection(pos_pred, yaw_pred, data_batch). pos_pred and yaw_pred have shape (B, N, T, 2) and (B, N, T, 1). They are all in agent coordinate. data_batch is a dictionary mentioned earlier. This function returns the projection of each vehicle predicted trajectory on its current lane in agent coordinate with shape (B, N, T, 3) where 3 represents (x, y, yaw). \
    5. get_left_lane_projection(pos_pred, yaw_pred, data_batch). It is similar to get_current_lane except it returns the left lane waypoints. If there is no left lane, the original trajectory will be returned. \
    6. get_right_lane_projection(pos_pred, yaw_pred, data_batch). It is similar to get_current_lane except it returns the right lane waypoints. If there is no left lane, the original trajectory will be returned."

    # Some concepts: \
    # 1. vehicles exist in a 2d-world which has a world coordinate. \
    # 2. each vehicle has its own agent coordinate in which its current position (x,y) is used as the center and its orientation (yaw) is the positive x axis facing ahead. A vehicle faces towards positive x-axis in its own agent coordinate and a vehicle's left direction is the positive y-axis in its own agent coordinate. \
    # 3. we use the word 'agent' and the word 'vehicle' interchangeably. \
    # 4. when a vehicle A follows another vehicle B, vehicle A should stay behind vehicle B facing a similar direction but not collide with vehicle B.\

    user_final_query = user_query + api_description 
    
    if context_description != "":
        user_final_query += context_description

    system_message = None

    from tbsim.utils.guidance_loss import AgentCollisionLoss, MapCollisionLoss
    agentcollision_source = getsource(AgentCollisionLoss)
    agentcollision_source = remove_comments(agentcollision_source)

    mapcollision_source = getsource(MapCollisionLoss)
    mapcollision_source = remove_comments(mapcollision_source)

    acclimit_source = getsource(AccLimitLoss)

    stayonleft_source = getsource(StayOnLeftLoss)

    few_shot_examples = [
        # (
        # "Generate a loss class such that all the vehicles should not collide with each other",
        # agentcollision_source
        # ),
        # (
        # "Generate a loss class such that all the vehicles should not drive off the road.",
        # mapcollision_source
        # ),
        (
        "Generate a loss class such that vehicle 1 should always drive with acceleration below acc_limit.",
        acclimit_source
        ),
        (
        "Generate a loss class such that vehicle 20 should always stay on the left side of vehicle 13.",
        stayonleft_source
        ),
    ]
    
    response = query_gpt(openai_key, user_final_query, system_message, few_shot_examples)
    return response['choices'][0]['message']['content']

from tbsim.utils.guidance_loss import GuidanceLoss
from tbsim.utils.trajdata_utils import select_agent_ind, transform_coord_agents_to_world, transform_coord_world_to_agent_i
import torch
class AccLimitLoss(GuidanceLoss):
    '''
    Vehicle with index target_ind should not drive with acceleration above acc_limit.
    '''
    def __init__(self, acc_limit, target_ind=1):
        super().__init__()
        self.acc_limit = acc_limit
        self.target_ind = target_ind
    def forward(self, x, data_batch, agt_mask=None):
        B, N, T, _ = x.shape
        if agt_mask is not None:
            x = x[agt_mask]
        # select the relevant agent with index self.target_ind
        # (B,N,T,6) -> (N,T,6)
        x_i = select_agent_ind(x, self.target_ind)
        # Select the acceleration
        # (N,T,6) -> (N,T)
        acc = x_i[..., 4]
        # Estimate the acceleration deviation from the limit
        # (N,T) -> (N,T)
        acc_dev = torch.abs(acc) - self.acc_limit
        # Clip the negative values to 0
        # (N,T) -> (N,T)
        acc_loss = torch.clip(acc_dev, min=0)
        # Take the mean over time
        # (N,T) -> (N) 
        loss = torch.mean(acc_loss, dim=-1)

        return loss
    
class StayOnLeftLoss(GuidanceLoss):
    '''
    Vehicle with index target_ind should always keep on the left side of vehicle with index ref_ind.
    '''
    def __init__(self, target_ind=20, ref_ind=13, decay_rate=0.9):
        super().__init__()
        self.target_ind = target_ind
        self.ref_ind = ref_ind
        self.decay_rate = decay_rate

    def forward(self, x, data_batch, agt_mask=None):
        B, N, T, _ = x.shape
        if agt_mask is not None:
            x = x[agt_mask]
        # Select positions
        # (B,N,T,6) -> (B,N,T,2)
        pos_pred = x[..., :2]
        # Select yaws
        # (B,N,T,6) -> (B,N,T,1)
        yaw_pred = x[..., 3:4]
        # convert prediction from the respective agent coordinates to the world coordinate
        # (B,N,T,2), (B,N,T,1), dict -> (B,N,T,2), (B,N,T,1)
        pos_pred_world, yaw_pred_world = transform_coord_agents_to_world(pos_pred, yaw_pred, data_batch)
        # convert prediction from the world coordinate to the agent self.ref_ind coordinate
        # (B,N,T,2), (B,N,T,1), dict, int -> (B,N,T,2), (B,N,T,1)
        pos_pred_in_ref_ind, _ = transform_coord_world_to_agent_i(pos_pred_world, yaw_pred_world, data_batch, self.ref_ind)

        # select the relevant agents with index self.target_ind in the agent self.ref_ind coordinate
        # (B, N, T, 2), int -> (N, T, 2)
        pos_pred_i_in_ref_ind = select_agent_ind(pos_pred_in_ref_ind, self.target_ind)
        # select the relevant agents with index self.target_ind in the agent self.ref_ind coordinate
        # (B, N, T, 2), int -> (N, T, 2)
        pos_pred_j_in_ref_ind = select_agent_ind(pos_pred_in_ref_ind, self.ref_ind)
        
        # Since we only care about the y-axis, we only need to compute the y-axis (rather than both x and y axis) deviation.
        # (N, T, 2), (N, T, 2) -> (N, T)
        pos_dev = pos_pred_j_in_ref_ind[...,1] - pos_pred_i_in_ref_ind[...,1] 
        
        # Clip the position deviation to 0 so that we only penalize the positive deviation
        # (N, T) -> (N, T)
        pos_loss = torch.clip(pos_dev, min=0)

        # penalize early steps more than later steps
        exp_weights = torch.tensor([self.decay_rate ** t for t in range(T)], device=x.device)
        exp_weights /= exp_weights.sum()
        pos_loss = pos_loss * exp_weights[None, :]
        # Take the mean over time
        # (N, T) -> (N)
        pos_loss = pos_loss.mean(-1)

        return pos_loss
