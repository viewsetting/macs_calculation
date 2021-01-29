
def calculate_linear_macs(parameter,seq_len=1):
    return parameter.numel()*seq_len

def calculate_attention_macs(hidden_dim,seq_len=1):
    """
    softmax(Q K_T /{sqrt(hidden_dim)}) V
    """
    return seq_len * hidden_dim * seq_len * 2 + seq_len**2