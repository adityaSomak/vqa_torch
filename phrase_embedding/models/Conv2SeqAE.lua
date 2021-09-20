require 'nngraph'
require 'torch'
local nn = require 'nn'
require 'cudnn'
require 'cunn'
--require 'rnn'
package.path = "/home/ASUAD/saditya1/Desktop/VQA/VQA_torch/phrase_embedding/models/wordrnn/RNN.lua;" .. package.path
local RNN = require 'RNN'


local ModelBuilder = torch.class('ModelBuilder')

local Model = {
  cellSizes = {256, 256}, -- Number of LSTM cells
  decLSTMs = {}
}

function ModelBuilder:createAutoencoder(w2v, opt)
  -- Create encoder
  --if opt.cudnn == 1 then

  --end

  local input = nn.Identity()()

  local lookup

  lookup = nn.LookupTable(opt.vocab_size, opt.vec_size)
  if opt.model_type == 'static' or opt.model_type == 'nonstatic' then
    lookup.weight:copy(w2v)
  else
    -- rand
    lookup.weight:uniform(-0.25, 0.25)
  end
  -- padding should always be 0
  lookup.weight[1]:zero()
  lookup = lookup(input)

  -- kernels is an array of kernel sizes
  local kernels = opt.kernels
  local layer1 = {}
  for i = 1, #kernels do
    local conv
    local conv_layer
    local max_time
    conv = cudnn.SpatialConvolution(1, opt.num_feat_maps, opt.vec_size, kernels[i])
    if opt.highway_conv_layers > 0 then
      -- Highway conv layers
      local highway_conv = HighwayConv.conv(opt.vec_size, opt.max_sent, kernels[i], opt.highway_conv_layers)
      conv_layer = nn.Reshape(opt.num_feat_maps, opt.max_sent-kernels[i]+1, true)(
        conv(nn.Reshape(1, opt.max_sent, opt.vec_size, true)(
        highway_conv(lookup))))
      max_time = nn.Max(3)(cudnn.ReLU()(conv_layer))
    else
      conv_layer = nn.Reshape(opt.num_feat_maps, opt.max_sent-kernels[i]+1, true)(
        conv(
        nn.Reshape(1, opt.max_sent, opt.vec_size, true)(
        lookup)))
      --max_time = nn.Max(3)(cudnn.ReLU()(conv_layer))
      max_time = nn.SpatialMaxPooling(10,1)(cudnn.ReLU()(conv_layer))
    end

    conv.weight:uniform(-0.01, 0.01)
    conv.bias:zero()
    conv.name = 'convolution'
    table.insert(layer1, max_time)
  end

  if opt.skip_kernel > 0 then
    -- skip kernel
    local kern_size = 5 -- fix for now
    local skip_conv = cudnn.SpatialConvolution(1, opt.num_feat_maps, opt.vec_size, kern_size)
    skip_conv.name = 'skip_conv'
    skip_conv.weight:uniform(-0.01, 0.01)
    -- skip center for now
    skip_conv.weight:select(3,3):zero()
    skip_conv.bias:zero()
    local skip_conv_layer = nn.Reshape(opt.num_feat_maps, opt.max_sent-kern_size+1, true)(skip_conv(nn.Reshape(1, opt.max_sent, opt.vec_size, true)(lookup)))
    table.insert(layer1, nn.Max(3)(cudnn.ReLU()(skip_conv_layer)))
  end

  local conv_layer_concat
  if #layer1 > 1 then
    conv_layer_concat = nn.JoinTable(2)(layer1)
  else
    conv_layer_concat = layer1[1]
  end

  local last_layer = conv_layer_concat

  -- Create decoder
  input_size_to_rnn = #kernels * opt.num_feat_maps;
  rnnoutputs = RNN.rnn(input_size_to_rnn, opt.vec_size, opt.max_sent, opt.dropout_p, last_layer)
  --local output = rnn(last_layer)

  -- Create autoencoder
  -- self.autoencoder =  nn.gModule({inputs}, {rnnoutputs})
  self.autoencoder =  nn.gModule({input}, {rnnoutputs})
end

return ModelBuilder
