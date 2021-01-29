from t3nsor import TensorTrain,TTLinear,TTEmbedding
from t3nsor.utils import auto_shape

def get_full_input_dim(cores):
    input_dim = 1
    for core in cores:
        input_dim *= core.shape[1]
    return input_dim

def calculate_TT_macs(cores,seq_len=1):
    core_num = len(cores)
    curr_shape = cores[0].shape
    curr_hid = get_full_input_dim(cores)
    data_shape = [seq_len,curr_shape[0],curr_shape[1],curr_hid//(curr_shape[0]*curr_shape[1])]
    
    macs = 0
    for core_idx in range(core_num):
        # batchsize * (left_rank * core_input_dim * input_left_dims * core_output_dim * right_rank )
        macs += data_shape[0] * data_shape[1]*data_shape[2] *data_shape[3] * curr_shape[2] *curr_shape[3]
        #update current input dim
        curr_hid = curr_hid * curr_shape[2] * curr_shape[3] // (curr_shape[0]*curr_shape[1])
        
        #get the next core shape list
        if core_idx+1 != core_num:
            next_shape = cores[core_idx+1].shape

        #reshape to adapt next core shape
        data_shape[3] = curr_hid // (next_shape[0]*next_shape[1])
        data_shape[1] = next_shape[0]
        data_shape[2] = next_shape[1]
         
        curr_shape = next_shape
    
    return macs 

def calculate_linear_macs(parameter,seq_len=1):
    return parameter.numel()*seq_len

def calculate_attention_macs(hidden_dim,seq_len=1):
    """
    softmax(Q K_T /{sqrt(hidden_dim)}) V
    """
    return seq_len * hidden_dim * seq_len * 2 + seq_len**2

if __name__ == '__main__':

    pass