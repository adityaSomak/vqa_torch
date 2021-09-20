require 'cudnn'
require 'cunn'


local function get_network()
  vocab_size= 60000;
  vec_size = 300;
  max_sent = 98;
  num_feat_maps=100;
  highway_conv_layers=0;
  skip_kernel=0;

  input = nn.Identity()()

  lookup = nn.LookupTable(vocab_size, vec_size)

  lookup.weight:uniform(-0.25, 0.25)

  -- padding should always be 0
  lookup.weight[1]:zero()
  lookup = lookup(input)

  -- kernels is an array of kernel sizes
  local kernels = {3,4,5}
  local layer1 = {}
  for i = 1, #kernels do
  local conv
  local conv_layer
  local max_time
  conv = cudnn.SpatialConvolution(1, num_feat_maps, vec_size, kernels[i])
  if highway_conv_layers > 0 then
    -- Highway conv layers
    local highway_conv = HighwayConv.conv(vec_size, max_sent, kernels[i], highway_conv_layers)
    conv_layer = nn.Reshape(num_feat_maps, max_sent-kernels[i]+1, true)(
      conv(nn.Reshape(1, max_sent, vec_size, true)(
      highway_conv(lookup))))
    max_time = nn.Max(3)(cudnn.ReLU()(conv_layer))
  else
    conv_layer = nn.Reshape(num_feat_maps, max_sent-kernels[i]+1, true)(
      conv(
      nn.Reshape(1, max_sent, vec_size, true)(
      lookup)))
    max_time = nn.Max(3)(cudnn.ReLU()(conv_layer))
  end

  conv.weight:uniform(-0.01, 0.01)
  conv.bias:zero()
  conv.name = 'convolution'
  table.insert(layer1, max_time)
  end

  if skip_kernel > 0 then
  	-- skip kernel
  	local kern_size = 5 -- fix for now
  	local skip_conv = cudnn.SpatialConvolution(1, num_feat_maps, vec_size, kern_size)
  	skip_conv.name = 'skip_conv'
  	skip_conv.weight:uniform(-0.01, 0.01)
  	-- skip center for now
  	skip_conv.weight:select(3,3):zero()
  	skip_conv.bias:zero()
  	local skip_conv_layer = nn.Reshape(num_feat_maps, max_sent-kern_size+1, true)(skip_conv(nn.Reshape(1, max_sent, vec_size, true)(lookup)))
  	table.insert(layer1, nn.Max(3)(cudnn.ReLU()(skip_conv_layer)))
  end

  local conv_layer_concat
  if #layer1 > 1 then
  conv_layer_concat = nn.JoinTable(2)(layer1)
  else
  conv_layer_concat = layer1[1]
  end

  local last_layer = conv_layer_concat

  nngraph.annotateNodes()
  autoencoder =  nn.gModule({input}, {last_layer})
  return autoencoder
end

mlp = get_network();
x = torch.Tensor{1,2,3,10,16,18,19};
mlp:updateOutput(x)
graph.dot(mlp.fg,'graph');