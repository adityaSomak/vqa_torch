-- Load dependencies
require 'hdf5'
local optim = require 'optim'
local gnuplot = require 'gnuplot'
local image = require 'image'
local cuda = pcall(require, 'cutorch') -- Use CUDA if available
local hasCudnn, cudnn = pcall(require, 'cudnn') -- Use cuDNN if available

-- Set up Torch
print('Setting up')
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(1)
if cuda then
  require 'cunn'
  cutorch.manualSeed(torch.random())
end


-- Choose model to train
local cmd = torch.CmdLine()
cmd:option('-model', 'Conv2SeqAE', 'Model: AE|SparseAE|ConvAE|UpconvAE|Conv2SeqAE')

cmd:text('Options')
cmd:option('-model_type', 'nonstatic', 'Model type. Options: rand (randomly initialized word embeddings), static (pre-trained embeddings from word2vec, static during learning), nonstatic (pre-trained embeddings, tuned during learning), multichannel (two embedding channels, one static and one nonstatic)')
cmd:option('-data', '', 'Training data and word2vec data')
cmd:option('-cudnn', 0, 'Use cudnn and GPUs if set to 1, otherwise set to 0')
cmd:option('-seed', 3435, 'random seed, set -1 for actual random')
cmd:option('-folds', 10, 'number of folds to use. If test set provided, folds=1. max 10')
cmd:option('-debug', 0, 'print debugging info including timing, confusions')
cmd:option('-gpuid', 0, 'GPU device id to use.')
cmd:option('-savefile', '', 'Name of output file, which will hold the trained model, model parameters, and training scores. Default filename is TIMESTAMP_results')
cmd:option('-zero_indexing', 0, 'If data is zero indexed')
cmd:option('-dump_feature_maps_file', '', 'Set file to dump feature maps of convolution')
cmd:text()

-- Training hyperparameters
cmd:option('-learningRate', 0.001, 'Learning rate')
cmd:option('-optimiser', 'adam', 'Optimiser')
cmd:option('-epochs', 10, 'Training epochs')
cmd:option('-batch_size', 150, 'Batch size for training')

-- Model hyperparameters
cmd:option('-num_feat_maps', 100, 'Number of feature maps after 1st convolution')
cmd:option('-kernels', '{3,4,5}', 'Kernel sizes of convolutions, table format.')
cmd:option('-skip_kernel', 0, 'Use skip kernel')
cmd:option('-dropout_p', 0.5, 'p for dropout')
cmd:option('-highway_mlp', 0, 'Number of highway MLP layers')
cmd:option('-highway_conv_layers', 0, 'Number of highway MLP layers')
cmd:option('-num_rnn_layers', 2, 'Number of RNN layers: 2/3')
cmd:text()
--opt = cmd:parse(arg)



function load_data()
  local train
  local test

  print('loading data...')
  assert(opt.data ~= '', 'must provide hdf5 datafile')
  local f = hdf5.open(opt.data, 'r')
  local w2v = f:read('w2v'):all()
  train = f:read('train'):all()
  -- train_label = f:read('train_label'):all()

  if f:read('dev'):dataspaceSize()[1] == 0 then
    opt.has_dev = 0
  else
    opt.has_dev = 1
    dev = f:read('dev'):all()
  end
  if f:read('test'):dataspaceSize()[1] == 0 then
    opt.has_test = 0
  else
    opt.has_test = 1
    test = f:read('test'):all()
  end
  print('data loaded!')

  return train, test, dev, w2v
end

function get_layer(model, name)
  local named_layer
  function get(layer)
    if layer.name == name or torch.typename(layer) == name then
      named_layer = layer
    end
  end

  model:apply(get)
  return named_layer
end

function main()
  opt = cmd:parse(arg)
  print(opt)
  -- Load Data
  local XTrain, XTest, XDev, w2v = load_data()
  local N = XTrain:size(1)
  opt.vocab_size = w2v:size(1)
  opt.vec_size = w2v:size(2)
  opt.max_sent = XTrain:size(2)
  print('vocab size: ', opt.vocab_size)
  print('vec size: ', opt.vec_size)
  print('Max sentence size: ', opt.max_sent)
  print('Training Data Size: ', N)
  if cuda then
    XTrain = XTrain:cuda()
    --XTest = XTest:cuda()
  end
  -- Retrieve kernels
  loadstring("opt.kernels = " .. opt.kernels)()

  local ModelBuilder = require ('models/' .. opt.model)
  local model_builder = ModelBuilder.new()
  local model = model_builder:createAutoencoder(w2v, opt)
  local autoencoder = model_builder.autoencoder
  if cuda then
    autoencoder:cuda()
    -- Use cuDNN if available
    if hasCudnn then
      cudnn.convert(autoencoder, cudnn)
    end
  end
  -- get layers
  local layers = {}
  layers['linear'] = get_layer(autoencoder, 'nn.Linear')
  layers['w2v'] = get_layer(autoencoder, 'nn.LookupTable')
  if opt.skip_kernel > 0 then
    layers['skip_conv'] = get_layer(autoencoder, 'skip_conv')
  end

  -- Get parameters
  local theta, gradTheta = autoencoder:getParameters()

  -- Create loss
  local criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())--nn.BCECriterion()
  if cuda then
    criterion:cuda()
  end

  -- Create optimiser function evaluation
  local x -- Minibatch
  --local xEmb -- Minibatch with word-embeddings
  local feval = function(params)
    if theta ~= params then
      theta:copy(params)
    end
    -- Zero gradients
    gradTheta:zero()

    -- Reconstruction phase
    -- Forward propagation
    local xHat = autoencoder:forward(x) -- Reconstruction
    local loss = criterion:forward(xHat, x)
    -- Backpropagation
    local gradLoss = criterion:backward(xHat, x)
    --gradLoss = torch.reshape(gradLoss,opt.batch_size,opt.max_sent,opt.vec_size);
    autoencoder:backward(x, gradLoss)

    -- Regularization phase
    if opt.model == 'Seq2SeqAE' or opt.model == 'Conv2SeqAE' then
      -- Clamp RNN gradients to prevent exploding gradients
      gradTheta:clamp(-10, 10)
    end

    if opt.model_type == 'static' then
      -- don't update embeddings for static model
      layers.w2v.gradWeight:zero()
    end
    return loss, gradTheta
  end

  -- Train
  print('Training')
  autoencoder:training()
  local optimParams = {learningRate = opt.learningRate}
  local __, loss
  local losses, advLosses = {}, {}
  lookupTable = get_layer(autoencoder, 'nn.LookupTable') 
  --nn.LookupTable(opt.vocab_size, opt.vec_size)
  --lookupTable.weight:copy(w2v)
  --lookupTable.weight[1]:zero()

  for epoch = 1, opt.epochs do
    print('Epoch ' .. epoch .. '/' .. opt.epochs)
    print('Batch size:' .. opt.batch_size)
    print('N: ' .. N .. ' batches: ' .. N/opt.batch_size)
    for n = 1, N, opt.batch_size do
      -- Get minibatch
      if n+opt.batch_size > N then
        break;
      end
      x = XTrain:narrow(1, n, opt.batch_size)
      -- Optimise
      __, loss = optim[opt.optimiser](feval, theta, optimParams)
      losses[#losses + 1] = loss[1]
      -- reset padding embedding to zero
      layers.w2v.weight[1]:zero()
      if opt.skip_kernel > 0 then
        -- keep skip kernel at zero
        layers.skip_conv.weight:select(3,3):zero()
      end
      print('Processing Batch Number ' .. n .. ' with loss: ' .. loss[1])
    end

    modelname = 'phrasemodel' .. epoch ..'.t7'
    torch.save(modelname , autoencoder);
    -- Plot training curve(s)
    local plots = {{'Autoencoder', torch.linspace(1, #losses, #losses), torch.Tensor(losses), '-'}}
    plotname = 'Training' .. epoch ..'.png'
    gnuplot.pngfigure(plotname)
    gnuplot.plot(table.unpack(plots))
    gnuplot.ylabel('Loss')
    gnuplot.xlabel('Batch #')
    gnuplot.plotflush()
  end

  --torch.save("phrasemodel.t7", autoencoder);
  if cuda then
    XTest = XTest:cuda()
  end
  -- Test
  print('Testing')
  x = XTest:narrow(1, 1, 100)
  local xHat
  autoencoder:evaluate()
  xHat = autoencoder:forward(x)
  -- Plot reconstructions
  -- image.save('Reconstructions.png', torch.cat(image.toDisplayTensor(x, 2, 10), image.toDisplayTensor(xHat, 2, 10), 1))
end

main()
