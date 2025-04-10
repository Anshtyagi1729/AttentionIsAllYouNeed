import torch
import torch.nn as nn 
import math
# same implementation as given in the paper "attention is all you need"
class InputEmbeddings(nn.Module):
    def __init__(self, d_model :int ,vocab_size :int):
        super().__init__()
        self.d_modal =d_model
        self.vocab_size=vocab_size
        self.embedding=nn.Embedding(vocab_size,d_model)
    def forward(self,x):
        return self.embedding(x)* math.sqrt(self.d_modal)
class PostitionalEncoding(nn.Module):
    def __init__(self,d_model:int,seq_len:int,dropout:float):
        super().__init__()
        self.d_model=d_model
        self.seq_len=seq_len
        self.dropout=nn.Dropout(dropout)
        #create the matrix of shape(seq_len,d_model)
        pe=torch.zeros(seq_len,d_model)
         #create the vector of the shape (seq_len)
        pos=torch.arrange(0,seq_len,dtype=torch.float).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        #apply the sin to even position and odd to the cosine 
        pe[:,0::2] = torch.sin(pos*div_term)
        pe[:,1::2] = torch.cos(pos*div_term)

        pe=pe.unsqueeze(0) #(1,sewq_len,d_model)
        self.register_buffer('pe',pe)
    def forward(self,x):
        x=x+ (self.pe[:,:x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)
class LayerNormalisation(nn.Module):
    def __init__(self, eps :float =10**-6):
        super().__init__()
        self.eps=eps
        self.alpha=nn.Parameter(torch.ones(1))#multiplied
        self.bias =nn.Parameter(torch.zeros(1)) # added
    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        std=x.std(dim=-1,keepdim=True)
        return self.alpha * (x-mean)/(std+self.eps) +self.bias
class FeedForward(nn.Module):
    def __init__(self,d_model:int ,d_ff:int , dropout:float):
        super().__init__()
        self.linear_1=nn.Linear(d_model,d_ff) #define the w1 and b1
        self.dropout=nn.Dropout(dropout)
        self.linear_2=nn.Linear(d_ff,d_model) #define the w2 and b2
    def forward(self,x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self,d_model:int,dropout:float,h:int):
        super().__init__()
        self.d_model=d_model
        self.h=h
        assert d_model % h ==0 , "d_model is not divisible by h"
        self.d_k=d_model//h
        self.w_q=nn.Linear(d_model,d_model)   #in the paper its q @ w_q or linear of w with W_q and this gives us the q'
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)
        self.w_o=nn.Linear(d_model,d_model)
        self.dropout=nn.Dropout(dropout)
    @staticmethod
    def attention(query,key,value,mask,dropout:nn.Dropout):
        d_k=query.shape[-1]
        attention_scores=(query @ key.transpose(-2,-1))/math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask==0,-1e9)
        attention_scores=attention_scores.softmax(dim=-1) #implementing the same attention formula from the paper
        if dropout is not None:
            attention_scores=dropout(attention_scores)
        return (attention_scores @ value),attention_scores    #returning the final attention and also the attention score for the visualisation
    


    def forward(self,q,k,v,mask):
        query=self.w_q(q) #size goes like this (batch,seq,dk)---(batch,seq,dk)
        key=self.w_k(k)#size goes like this (batch,seq,dk)---(batch,seq,dk)
        value=self.w_v(v)#size goes like this (batch,seq,dk)---(batch,seq,dk)
        #shape --> (batch,seq,d_model) turn this and make it into dk lists dk=d_model//h
        query=query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        #(batch,seq,d_model)->(batch,h,seq,d_k)
        key=query.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        value=value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)
        x,self.attention_score=MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)
        #redo the transpose we did earlier and concat the tensor of many heads
        x=x.transpose(1,2).contigous().view(x.shape[0],-1,self.h*self.d_k)
        #(batch,seq,d_model)-->(batch,seq,d_modal)

        return self.w_o(x)
    
#lets implement the residual connection in the encoder and decoder
# residual does the skip connections the x+sublayer(x)
class ResidualConnection(nn.Module):
    def __init__(self,  dropout:float):
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.norm=LayerNormalisation()
    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
#now let me define the encoder which will contain these above things plus the residual connections
class EncoderBlock(nn.Module):
    def __init__(self,self_attention_block:MultiHeadAttentionBlock,feed_forward_block:FeedForward,dropout:float):
        super().__init__()
        self.self_attention_block=self_attention_block
        self.feed_forward_block=feed_forward_block
        self.residual_connections=nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    #we define the residual connection and form the module for the transformer block like this 
    def forward(self,x,src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,src_mask))
        x= self.residual_connections[0](x,self.feed_forward_block)
        return x
#we make a encoder using the encoder block and this will lead to the formation of our encoder which will be 
# used n times by our model  
class Encoder(nn.Module):
    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalisation()
    def forward(self,x,mask):
        for layer in self.layers:
            x=layer(x,mask)
        return self.norm(x)
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block:MultiHeadAttentionBlock,cross_attention_block : MultiHeadAttentionBlock ,feed_forward_block:FeedForward,dropout:float):
        super().__init__()
        self.self_attention_block=self_attention_block
        self.cross_attention_block=cross_attention_block
        self.feed_forward_block=feed_forward_block
        self.residual_connections=nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    def forward(self,x,encoder_output,src_mask,tgt_mask):
        x=self.residual_connections[0](x, lambda x:self.self_attention_block(x,x,x,tgt_mask))
        x=self.residual_connections[1](x,lambda x:self.self_attention_block(x,encoder_output,encoder_output,src_mask))
        x=self.residual_connections[2](x,lambda x:self.feed_forward_block)
        return x
class Decoder(nn.Module):
    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalisation()
    def forward(self,x,encoder_output,src_mask,tgt_mask):
        for layer in self.layers:
            x=layer(x,encoder_output,src_mask,tgt_mask)
        return self.norm(x)
#we here define the linear layer for the transformer architecture
class ProjectionLayer(nn.Module):
    def __init__(self, d_model:int ,vocab_size :int):
        super().__init__()
        self.proj=nn.Linear(d_model,vocab_size)
    def forward(self,x):
        return torch.log_softmax(self.proj(x),dim=-1)
class Transformer(nn.Module):
    def __init__(self,encoder:Encoder,decoder:Decoder,src_embed:InputEmbeddings,tgt_embed:InputEmbeddings,src_pos:PostitionalEncoding,tgt_pos:PostitionalEncoding,projection_layer:PostitionalEncoding):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.src_embed=src_embed
        self.tgt_embed=tgt_embed
        self.src_pos=src_pos
        self.tgt_pos=tgt_pos
        self.projection_layer=projection_layer
    def encoder(self,src,src_mask):
        src=self.src_embed(src)
        src=self.src_embed(src)
        return self.encoder(src,src_mask)
    def decode(self,encoder_output,src_mask,tgt_mask):
        tgt=self.tgt_embed(tgt)
        tgt=self.src_pos(tgt)
        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)
    def project(self,x):
        return self.projection_layer(x)
def build_transformer(src_vocab_size:int,tgt_vocab_size:int,src_seq_len:int,tgt_seq_len:int,d_model:int =512,N:int =6,h:int =8,dropout:float =0.1,d_ff:int =2048)->Transformer:

#creating the embedding 
    src_embed=InputEmbeddings(d_model,src_vocab_size)
    tgt_embed=InputEmbeddings(d_model,tgt_vocab_size)
#create the positional encoding 
    src_pos=PostitionalEncoding(d_model,src_seq_len,dropout)
    tgt_pos=PostitionalEncoding(d_model,tgt_seq_len,dropout)
#create the encoder block
    encoder_blocks=[]
    for _ in range(N):
        encoder_self_attention_block=MultiHeadAttentionBlock(d_model,dropout)
        feed_forward_block=FeedForward(d_model,d_ff,dropout)
        encoder_block=EncoderBlock(encoder_self_attention_block,feed_forward_block,dropout)
        encoder_blocks.append(encoder_block)

#create the decoder block
    decoder_blocks=[]
    for _ in range(N):
        decoder_self_attention_block=MultiHeadAttentionBlock(d_model,h,dropout)
        decoder_cross_attention_block=MultiHeadAttentionBlock(d_model,h,dropout)
        feed_forward_block=FeedForward(d_model,d_ff,dropout)
        decoder_block=DecoderBlock(decoder_self_attention_block,decoder_cross_attention_block,feed_forward_block,dropout)
        decoder_blocks.append(decoder_block)

#creating the decoder and the encoder 
    encoder=Encoder(nn.ModuleList(encoder_blocks))
    decoder=Decoder(nn.ModuleList(decoder_blocks))
#creating the projection layer
    projection_layer=ProjectionLayer(d_model,tgt_vocab_size)
    #create the main transformer
    Transformer=Transformer(encoder,decoder,src_embed,tgt_embed,src_pos,tgt_pos,projection_layer)
    #init the params
    for p in Transformer.parameters():
        if p.dim()>1:
            nn.init.xavier_normal_(p)
        
    return Transformer







        
        




